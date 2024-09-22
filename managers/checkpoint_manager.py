# managers/checkpoint_manager.py

import torch
import os
from config.config import CHECKPOINT_DIR

class CheckpointManager:
    def __init__(self, latest_checkpoint_filename='pong_ddqn_checkpoint_1474.pth'):
        self.checkpoint_dir = CHECKPOINT_DIR
        self.latest_checkpoint = os.path.join(self.checkpoint_dir, latest_checkpoint_filename)

    def load_checkpoint(self, agent_manager):
        if os.path.exists(self.latest_checkpoint):
            try:
                checkpoint = torch.load(self.latest_checkpoint, map_location=agent_manager.device)
                agent_manager.load_model(checkpoint)
                episode = checkpoint.get('episode', 0)
                print(f"Checkpoint loaded, resuming from episode {episode}")
                return agent_manager, episode
            except Exception as e:
                print(f"Error loading checkpoint '{self.latest_checkpoint}': {e}")
        else:
            print("No checkpoint found, starting from scratch")
        return agent_manager, 0

    def save_checkpoint(self, agent_manager, episode, additional_info=None):
        checkpoint = agent_manager.save_model_state()
        checkpoint.update({
            'episode': episode,
            'metrics_file': additional_info.get('metrics_file') if additional_info else None
        })
        # Ensure the metrics_file name matches MetricsManager
        checkpoint_path = os.path.join(self.checkpoint_dir, f'pong_ddqn_checkpoint_{episode}.pth')
        try:
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, self.latest_checkpoint)
            print(f"Checkpoint saved at episode {episode} -> {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")