# agents/ddqn_agent.py

import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from models.dqn import DQN
from preprocess.frame_stack import FrameStack
from memory.replay_memory import ReplayMemory
from config.config import (
    DEVICE, ALPHA, MEMORY_CAPACITY, EPSILON_MIN, EPSILON_DECAY, 
    BATCH_SIZE, GAMMA
)
from memory.replay_memory import Experience


class DDQNAgent:
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.epsilon = 1.0  # Start with full exploration

        self.policy_net = DQN(input_shape, n_actions).to(DEVICE)
        self.target_net = DQN(input_shape, n_actions).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        self.target_net.eval()  # Set target net to eval mode

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=ALPHA)
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        self.frame_stack = FrameStack()

    # Select action using epsilon-greedy policy
    def select_action(self, state):
        if random.random() > self.epsilon:  # Exploit
            with torch.no_grad():  # Disable gradient computation
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # Convert state to tensor
                self.policy_net.eval()  # Set network to eval mode
                q_values = self.policy_net(state)  # Get Q-values from policy_net
                self.policy_net.train()  # Set back to training mode
                return q_values.max(1)[1].item()  # Return action with max Q-value
        else:  # Explore
            return random.randint(0, self.n_actions - 1)
        
    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:  # Not enough experiences
            return 0, 0

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.get_batch(BATCH_SIZE)

        # Get Q-values for the current state (from policy_net)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: select action from policy_net, evaluate from target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + GAMMA * next_state_values * (1 - done_batch)  # Bellman equation
        
        # Compute Huber loss
        loss = nn.SmoothL1Loss()(q_values.squeeze(1), expected_q_values)

        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients 
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Clip gradients
        self.optimizer.step()  # Update the policy_net 

        return loss.item(), q_values.max().item()