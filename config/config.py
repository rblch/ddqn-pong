# config/config.py

import os
import torch

# Constants and Hyperparameters
ENVIRONMENT = "ALE/Pong-v5"
FRAME_SHAPE = (64, 84)
NUM_FRAMES = 4  # Number of frames stacked together
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 1e-4  # Learning rate
MEMORY_CAPACITY = 20000  # Size of the replay memory
EPSILON_MIN = 0.05  # Epsilon decays up to this threshold 
EPSILON_DECAY = 0.995 
BATCH_SIZE = 32  # Number of experiences to sample for training step
GAMMA = 0.99  # Discount factor for future rewards
NUM_EPISODES = 2000
MAX_STEPS = 10000
MIN_MEMORY_LEN = 10000  # Minimum number of experiences before starting training
TARGET_UPDATE = 2  # Number of episodes between target network updates
SAVE_MODEL_INTERVAL = 25  # Number of episodes between saving model checkpoints

# Directories for saving checkpoints and metrics
CHECKPOINT_DIR = "checkpoints"
METRICS_FILE = "training_metrics.json"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
