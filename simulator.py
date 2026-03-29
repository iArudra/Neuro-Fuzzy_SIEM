import time
import requests
import logging

LOG_FILE = r"d:\College\PROJECTS\SIEM\dataset\HDFS.log"
API_URL = "http://127.0.0.1:8000/ingest"

def simulate_stream(batch_size=50, delay=1.0):
    print(f"Starting Log Simulation... Sending {batch_size} logs every {delay}s")
    try:
        with open(LOG_FILE, "r") as f:
            batch = []
            for line in f:
                batch.append(line.strip())
                if len(batch) >= batch_size:
                    try:
                        res = requests.post(API_URL, json={"logs": batch})
                        print(f"Sent {len(batch)} logs. Server responded: HTTP {res.status_code}")
                    except Exception as e:
                        print(f"Failed to send logs: {e}. Is the server running?")
                    
                    batch = []
                    time.sleep(delay)
                    
            if batch:
                requests.post(API_URL, json={"logs": batch})
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    except FileNotFoundError:
        print(f"Log file not found at {LOG_FILE}")

if __name__ == "__main__":
    simulate_stream(batch_size=50, delay=1.0)
