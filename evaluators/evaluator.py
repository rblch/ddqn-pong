# evaluators/evaluator.py

import torch
import numpy as np
from managers.environment_manager import EnvironmentManager
from managers.agent_manager import AgentManager
from preprocess.preprocess import Preprocess
from config.config import FRAME_SHAPE, NUM_FRAMES
from agents.ddqn_agent import DDQNAgent
from memory.replay_memory import ReplayMemory
from collections import deque
import matplotlib.pyplot as plt
import json
import os

class Evaluator:
    def __init__(self, agent_checkpoint_path, num_episodes=100, render=False):
        self.env_manager = EnvironmentManager()
        self.preprocess = Preprocess()
        self.num_episodes = num_episodes
        self.render = render

        # Initialize the trained agent
        self.agent_manager = AgentManager(input_shape=(NUM_FRAMES, *FRAME_SHAPE), n_actions=self.env_manager.n_actions)
        self.load_trained_agent(agent_checkpoint_path)
    
    def load_trained_agent(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Use CPU for evaluation
            self.agent_manager.load_model(checkpoint)
            self.agent_manager.agent.policy_net.eval()  # Set to evaluation mode
            print(f"Loaded trained agent from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
    
    def run_episode(self, agent, render=False):
        obs, _ = self.env_manager.reset()
        state = self.preprocess.preprocess(obs)
        frame_stack = deque(maxlen=NUM_FRAMES)
        
        # Initialize frame stack with the first state
        for _ in range(NUM_FRAMES):
            frame_stack.append(state)
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            if render:
                self.env_manager.env.render()
            
            # Prepare the stacked state
            stacked_state = np.stack(frame_stack, axis=0)
            
            if isinstance(agent, AgentManager):
                # Trained agent selects action
                action = agent.select_action(stacked_state)
            else:
                # Random agent selects action
                action = np.random.randint(0, self.env_manager.n_actions)
            
            obs, reward, terminated, truncated, _ = self.env_manager.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Preprocess next state and update frame stack
            next_state = self.preprocess.preprocess(obs)
            frame_stack.append(next_state)
        
        return episode_reward, steps
    
    def evaluate_agent(self, render=False):
        rewards = []
        steps_list = []
        for episode in range(1, self.num_episodes + 1):
            reward, steps = self.run_episode(self.agent_manager, render=render)
            rewards.append(reward)
            steps_list.append(steps)
            print(f"Trained Agent - Episode {episode}: Reward = {reward}, Steps = {steps}")
        return rewards, steps_list
    
    def evaluate_random_agent(self, render=False):
        rewards = []
        steps_list = []
        for episode in range(1, self.num_episodes + 1):
            reward, steps = self.run_episode(self.env_manager, render=render)
            rewards.append(reward)
            steps_list.append(steps)
            print(f"Random Agent - Episode {episode}: Reward = {reward}, Steps = {steps}")
        return rewards, steps_list
    
    def visualize_results(self, trained_rewards, random_rewards):
        plt.figure(figsize=(12, 6))

        # Plot average rewards
        plt.subplot(1, 2, 1)
        plt.hist(trained_rewards, bins=20, alpha=0.7, label='Trained Agent')
        plt.hist(random_rewards, bins=20, alpha=0.7, label='Random Agent')
        plt.xlabel('Episode Return')
        plt.ylabel('Frequency')
        plt.title('Return Distribution')
        plt.legend()

        # Plot cumulative rewards
        plt.subplot(1, 2, 2)
        plt.boxplot([trained_rewards, random_rewards], labels=['Trained Agent', 'Random Agent'])
        plt.ylabel('Episode Return')
        plt.title('Return Comparison')

        plt.tight_layout()
        plt.show()
    
    def save_results(self, trained_rewards, random_rewards, save_path='evaluation_results.json'):
        results = {
            'trained_agent': {
                'rewards': trained_rewards,
                'average_reward': np.mean(trained_rewards),
                'std_reward': np.std(trained_rewards)
            },
            'random_agent': {
                'rewards': random_rewards,
                'average_reward': np.mean(random_rewards),
                'std_reward': np.std(random_rewards)
            }
        }
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation results saved to {save_path}")
    
    def close(self):
        self.env_manager.close()