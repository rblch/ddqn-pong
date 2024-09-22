# managers/metrics_manager.py

import json
import os
from config.config import METRICS_FILE

class MetricsManager:
    def __init__(self):
        self.metrics_file = METRICS_FILE
        self.metrics = self.load_metrics()

    def load_metrics(self):
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
                    print(f"Loaded {len(metrics)} episodes from metrics file")
                    return metrics
            except json.JSONDecodeError:
                print("Metrics file is empty or corrupted - starting from scratch")
        return []

    def save_metrics(self, episode_metrics):
        self.metrics.append(episode_metrics)
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def truncate_metrics(self, episode):
        if len(self.metrics) < episode:
            print("Metrics file has fewer episodes than the checkpoint - Adjusting metrics list.")
            self.metrics = self.metrics[:episode]
