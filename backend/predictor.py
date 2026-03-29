import sys
import os
import torch
import numpy as np
from collections import defaultdict, deque

# Allow importing from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'model_training'))

from model import NeuroFuzzySIEM
from preprocessing import parse_log_line, load_templates
from inference import load_inference_artifacts, get_alert_level

class StatefulPredictor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.raw_logs_window = deque(maxlen=window_size)
        self.parsed_logs_window = deque(maxlen=window_size)
        
        # Load artifacts
        print("Loading templates and artifacts...")
        self.template_patterns, self.templates_df = load_templates()
        self.scaler, self.vocab_size = load_inference_artifacts()
        
        # Setup event mapping
        all_event_ids = ['E_UNMATCHED', 'PAD'] + list(self.templates_df['EventId'])
        self.event_id_map = {eid: idx for idx, eid in enumerate(all_event_ids)}
        
        # Load model
        print("Loading Neuro-Fuzzy SIEM model...")
        model_path = os.path.join(parent_dir, "neuro_fuzzy_siem.pth")
        self.model = NeuroFuzzySIEM(vocab_size=self.vocab_size, num_numeric_features=5)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def ingest_log(self, raw_log_line):
        parsed = parse_log_line(raw_log_line, self.template_patterns)
        if not parsed:
            return None
        
        self.raw_logs_window.append(raw_log_line)
        self.parsed_logs_window.append(parsed)
        
        # We need a minimum amount of valid logs to pad
        if len(self.parsed_logs_window) == 0:
            return None
            
        if len(self.parsed_logs_window) < self.window_size:
            # Pad
            padded = [{'datetime': self.parsed_logs_window[0]['datetime'], 'event_id': 'PAD', 'is_error': 0} for _ in range(self.window_size - len(self.parsed_logs_window))]
            window = padded + list(self.parsed_logs_window)
        else:
            window = list(self.parsed_logs_window)
            
        # Extract features
        event_seq = [self.event_id_map.get(lg['event_id'], 0) for lg in window]
        
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
        seq_length = len(self.parsed_logs_window)
        
        num_feats = [mean_ts, max_ts, error_count, unique_events, seq_length]
        
        num_feats_scaled = self.scaler.transform([num_feats])[0]
        
        x_cat = torch.tensor([event_seq], dtype=torch.long)
        x_num = torch.tensor([num_feats_scaled], dtype=torch.float32)
        
        with torch.no_grad():
            risk_score = self.model(x_cat, x_num).item()
            
        alert_level = get_alert_level(risk_score)
        
        return {
            "parsed": parsed,
            "risk_score": risk_score,
            "alert_level": alert_level,
            "window_logs": list(self.raw_logs_window)
        }
