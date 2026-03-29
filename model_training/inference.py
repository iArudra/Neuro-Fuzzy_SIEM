import os
import pickle
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from model import NeuroFuzzySIEM
from preprocessing import parse_log_line, load_templates

MODEL_FILE = r"d:\College\PROJECTS\SIEM\neuro_fuzzy_siem.pth"
DATA_FILE = r"d:\College\PROJECTS\SIEM\processed_data.pkl"
WINDOW_SIZE = 10

def load_inference_artifacts():
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    scaler = data['scaler']
    vocab_size = data['vocab_size']
    return scaler, vocab_size

def get_alert_level(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Critical"

def preprocess_sequence(raw_logs, template_patterns, scaler, vocab_size):
    # Dummy mapping for event IDs based on vocab (assume 0 is PAD, 1 is E_UNMATCHED, 2+ are templates)
    # We load templates_df in load_templates() to get the exact mapping.
    _, templates_df = load_templates()
    all_event_ids = ['E_UNMATCHED', 'PAD'] + list(templates_df['EventId'])
    event_id_map = {eid: idx for idx, eid in enumerate(all_event_ids)}
    
    parsed_logs = []
    for line in raw_logs:
        parsed = parse_log_line(line, template_patterns)
        if parsed:
            parsed_logs.append(parsed)
            
    if not parsed_logs:
        raise ValueError("No valid logs parsed.")
        
    parsed_logs.sort(key=lambda x: x['datetime'])
    seq_length = len(parsed_logs)
    
    # Take the last WINDOW_SIZE logs, pad if necessary
    if seq_length < WINDOW_SIZE:
        padded_logs = [{'datetime': parsed_logs[0]['datetime'], 'event_id': 'PAD', 'is_error': 0} for _ in range(WINDOW_SIZE - seq_length)] + parsed_logs
        window = padded_logs
    else:
        window = parsed_logs[-WINDOW_SIZE:]
        
    # Categorical features
    event_seq = [event_id_map.get(lg['event_id'], 0) for lg in window]
    
    # Numerical features
    timestamps = [lg['datetime'] for lg in window]
    ts_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
    ts_diffs.insert(0, 0.0)
    
    error_count = sum(lg['is_error'] for lg in window)
    
    freq_dict = defaultdict(int)
    for lg in window:
        if lg['event_id'] != 'PAD':
            freq_dict[lg['event_id']] += 1
            
    mean_ts = np.mean(ts_diffs)
    max_ts = np.max(ts_diffs)
    unique_events = len(freq_dict)
    
    num_feats = [mean_ts, max_ts, error_count, unique_events, seq_length]
    
    # Scale numerical
    num_feats_scaled = scaler.transform([num_feats])[0]
    
    x_cat = torch.tensor([event_seq], dtype=torch.long)
    x_num = torch.tensor([num_feats_scaled], dtype=torch.float32)
    
    return x_cat, x_num

def run_inference(raw_logs):
    print("Loading templates and artifacts...")
    template_patterns, _ = load_templates()
    scaler, vocab_size = load_inference_artifacts()
    
    print("Preprocessing input log sequence...")
    x_cat, x_num = preprocess_sequence(raw_logs, template_patterns, scaler, vocab_size)
    
    print("Loading Neuro-Fuzzy SIEM model...")
    model = NeuroFuzzySIEM(vocab_size=vocab_size, num_numeric_features=x_num.shape[1])
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        risk_score = model(x_cat, x_num).item()
        
    alert_level = get_alert_level(risk_score)
    
    print("\n" + "="*40)
    print("        INFERENCE RESULT        ")
    print("="*40)
    print(f"Log Sequence Length : {len(raw_logs)}")
    print(f"Computed Risk Score : {risk_score:.2f}")
    print(f"Alert Level         : {alert_level}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Example raw logs (mix of normal and error statements)
    sample_logs = [
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
        "081109 203616 148 INFO dfs.DataNode$PacketResponder: Received block blk_38865049064139660 of size 67108864 from /10.250.19.102",
        "081109 203617 148 WARN dfs.DataNode$DataXceiver: Got exception while serving blk_38865049064139660 to /10.250.19.102",
        "081109 203618 148 ERROR dfs.DataNode$DataXceiver: Exception in receiveBlock for block blk_38865049064139660",
    ]
    
    run_inference(sample_logs)
