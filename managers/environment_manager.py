# managers/environment_manager.py
import gymnasium as gym
from config.config import ENVIRONMENT

class EnvironmentManager:
    def __init__(self):
        self.env = gym.make(ENVIRONMENT)
        self.full_action_space = self.env.action_space.n
        self.n_actions = 3  # NOOP, UP, DOWN

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")
        
        gym_action = {
            0: 0,  # NOOP
            1: 2,  # UP 
            2: 3   # DOWN 
        }[action]
        
        return self.env.step(gym_action)

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.n_actions