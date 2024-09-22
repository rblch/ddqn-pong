# managers/agent_manager.py

from agents.ddqn_agent import DDQNAgent
from config.config import DEVICE

class AgentManager:
    def __init__(self, input_shape, n_actions):
        self.agent = DDQNAgent(input_shape, n_actions)
        self.device = DEVICE

    def select_action(self, state):
        return self.agent.select_action(state)

    def optimize_model(self):
        return self.agent.optimize_model()

    def update_epsilon(self):
        self.agent.update_epsilon()

    def load_model(self, checkpoint):
        self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.epsilon = checkpoint['epsilon']

    def save_model_state(self):
        return {
            'model_state_dict': self.agent.policy_net.state_dict(),
            'target_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon
        }
