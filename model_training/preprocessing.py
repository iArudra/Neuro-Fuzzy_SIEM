import os
import re
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"d:\College\PROJECTS\SIEM\dataset"
LOG_FILE = os.path.join(DATA_DIR, "HDFS.log")
TEMPLATE_FILE = os.path.join(DATA_DIR, "preprocessed", "HDFS.log_templates.csv")
LABEL_FILE = os.path.join(DATA_DIR, "preprocessed", "anomaly_label.csv")
OUTPUT_FILE = r"d:\College\PROJECTS\SIEM\processed_data.pkl"

MAX_LINES = 1000000  # Process only 1M lines to avoid memory issues
WINDOW_SIZE = 10

def load_templates():
    templates_df = pd.read_csv(TEMPLATE_FILE)
    template_patterns = []
    for _, row in templates_df.iterrows():
        event_id = row['EventId']
        template_str = row['EventTemplate']
        # Convert [*] to (.*)
        # Escape regex special characters except [*]
        escaped_str = re.escape(template_str).replace(r'\[\*\]', r'(.*)')
        pattern = re.compile(escaped_str)
        template_patterns.append((event_id, pattern))
    return template_patterns, templates_df

def match_event_id(content, template_patterns):
    for event_id, pattern in template_patterns:
        if pattern.search(content):
            return event_id
    return "E_UNMATCHED"

def parse_log_line(line, template_patterns):
    # Example line: 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906...
    # regex to extract Date, Time, Pid, Level, Component, Content
    match = re.match(r'^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+(.*?):\s+(.*)$', line.strip())
    if not match:
        return None
    date_str, time_str, pid, level, component, content = match.groups()
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%y%m%d %H%M%S")
    except ValueError:
        return None
        
    # Extract BlockId
    block_match = re.search(r'(blk_-?\d+)', content)
    block_id = block_match.group(1) if block_match else "UNKNOWN_BLOCK"
    
    # Check for error keywords
    is_error = 1 if re.search(r'(?i)(exception|error|failed|timeout|interrupted)', content) or level in ['WARN', 'ERROR', 'FATAL'] else 0

    event_id = match_event_id(content, template_patterns)
    
    return {
        'datetime': dt,
        'block_id': block_id,
        'event_id': event_id,
        'is_error': is_error
    }

def main():
    print("Loading templates...")
    template_patterns, templates_df = load_templates()
    
    print("Loading anomaly labels...")
    labels_df = pd.read_csv(LABEL_FILE)
    label_dict = dict(zip(labels_df['BlockId'], labels_df['Label']))
    
    print(f"Parsing logs (max {MAX_LINES} lines)...")
    blocks_data = defaultdict(list)
    
    with open(LOG_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
            if i > 0 and i % 100000 == 0:
                print(f"Processed {i} lines...")
            
            parsed = parse_log_line(line, template_patterns)
            if parsed and parsed['block_id'] != "UNKNOWN_BLOCK":
                blocks_data[parsed['block_id']].append(parsed)

    print(f"Extracted {len(blocks_data)} blocks. Generating sliding windows...")
    
    # Generate event_id to integer mapping
    all_event_ids = ['E_UNMATCHED', 'PAD'] + list(templates_df['EventId'])
    event_id_map = {eid: idx for idx, eid in enumerate(all_event_ids)}
    
    X_cat, X_num, y = [], [], []
    
    for block_id, logs in blocks_data.items():
        # Sort logs by datetime
        logs.sort(key=lambda x: x['datetime'])
        seq_length = len(logs)
        
        # Windows
        if seq_length < WINDOW_SIZE:
            # Pad sequence
            padded_logs = [{'datetime': logs[0]['datetime'], 'event_id': 'PAD', 'is_error': 0} for _ in range(WINDOW_SIZE - seq_length)] + logs
            windows = [padded_logs]
        else:
            windows = [logs[i:i+WINDOW_SIZE] for i in range(seq_length - WINDOW_SIZE + 1)]
            
        label_str = label_dict.get(block_id, "Normal")
        label_val = 1.0 if label_str == "Anomaly" else 0.0
        
        for w in windows:
            # Categorical features: sequence of event ids
            event_seq = [event_id_map.get(lg['event_id'], 0) for lg in w]
            
            # Numerical features
            timestamps = [lg['datetime'] for lg in w]
            ts_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            ts_diffs.insert(0, 0.0) # Pad the first diff
            
            error_count = sum(lg['is_error'] for lg in w)
            
            # Frequencies of events
            freq_dict = defaultdict(int)
            for lg in w:
                if lg['event_id'] != 'PAD':
                    freq_dict[lg['event_id']] += 1
            
            # Features: [mean_ts_diff, max_ts_diff, error_count, unique_events, seq_length]
            mean_ts = np.mean(ts_diffs)
            max_ts = np.max(ts_diffs)
            unique_events = len(freq_dict)
            
            num_feats = [mean_ts, max_ts, error_count, unique_events, seq_length]
            
            X_cat.append(event_seq)
            X_num.append(num_feats)
            y.append(label_val)

    print("Scaling numerical features...")
    X_cat = np.array(X_cat)
    X_num = np.array(X_num)
    y = np.array(y, dtype=np.float32)
    
    scaler = StandardScaler()
    if len(X_num) > 0:
        X_num = scaler.fit_transform(X_num)
        
    print(f"Final data shape: X_cat: {X_cat.shape}, X_num: {X_num.shape}, y: {y.shape}")
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'X_cat': X_cat,
            'X_num': X_num,
            'y': y,
            'vocab_size': len(event_id_map),
            'scaler': scaler
        }, f)
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
