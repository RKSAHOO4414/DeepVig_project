# utils/logger.py
"""Lightweight logger with wandb and CSV fallback."""

import os
import csv
import wandb


class WandbLogger:
    def __init__(self, project='DeepViG', name=None, use_wandb=True):
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                wandb.init(project=project, name=name)
                self.wandb = wandb
            except Exception as e:
                print(f'wandb init failed: {e}. Falling back to CSV.')
                self.use_wandb = False

        self.csv_path = os.path.join('logs', f'{name or "run"}.csv')
        os.makedirs('logs', exist_ok=True)
        self._csv_header_written = False

    def log(self, metrics: dict):
        if self.use_wandb:
            wandb.log(metrics)

        # CSV fallback
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(metrics)

    def finish(self):
        if self.use_wandb:
            wandb.finish()