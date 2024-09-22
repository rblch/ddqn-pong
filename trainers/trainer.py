# trainers/trainer.py

from managers.environment_manager import EnvironmentManager
from managers.agent_manager import AgentManager
from managers.metrics_manager import MetricsManager
from managers.checkpoint_manager import CheckpointManager
from preprocess.preprocess import Preprocess, FrameStack
from config.config import (
    FRAME_SHAPE, NUM_FRAMES, NUM_EPISODES, MAX_STEPS, 
    MIN_MEMORY_LEN, TARGET_UPDATE, SAVE_MODEL_INTERVAL
)

class Trainer:
    def __init__(self, config):
        self.env_manager = EnvironmentManager()
        self.agent_manager = AgentManager(input_shape=(NUM_FRAMES, *FRAME_SHAPE), n_actions=self.env_manager.n_actions)
        self.metrics_manager = MetricsManager()
        self.checkpoint_manager = CheckpointManager()
        self.preprocess = Preprocess()
        self.start_episode = 0
        self.load_existing_checkpoint()

    def load_existing_checkpoint(self):
        self.agent_manager, loaded_episode = self.checkpoint_manager.load_checkpoint(self.agent_manager)
        self.start_episode = loaded_episode + 1
        self.metrics_manager.truncate_metrics(self.start_episode)
        if loaded_episode > 0:
            print(f"Resuming training from episode {self.start_episode}")
        else:
            print("Starting training from scratch")

    def save_checkpoint(self, episode):
        self.checkpoint_manager.save_checkpoint(
            self.agent_manager,
            episode,
            additional_info={'metrics_file': 'training_metrics.json'}
        )

    def train(self):
        for episode in range(self.start_episode, NUM_EPISODES):
            obs, _ = self.env_manager.reset()
            state = self.preprocess.preprocess(obs)

            # Reset FrameStack at the start of each episode
            self.agent_manager.agent.frame_stack = FrameStack()

            self.agent_manager.agent.frame_stack.push(state)

            episode_reward = 0
            episode_loss = 0
            episode_q_value = 0
            optimization_step = 0

            for step in range(MAX_STEPS):
                state = self.agent_manager.agent.frame_stack.get_stacked_frames()
                action = self.agent_manager.select_action(state)

                obs, reward, terminated, truncated, _ = self.env_manager.step(action)
                done = terminated or truncated

                next_state = self.preprocess.preprocess(obs)
                self.agent_manager.agent.frame_stack.push(next_state)
                next_state_stacked = self.agent_manager.agent.frame_stack.get_stacked_frames()

                self.agent_manager.agent.memory.push(state, action, reward, next_state_stacked, done)

                episode_reward += reward

                if len(self.agent_manager.agent.memory) >= MIN_MEMORY_LEN:
                    loss, max_q = self.agent_manager.optimize_model()
                    episode_loss += loss
                    episode_q_value += max_q
                    optimization_step += 1

                if done:
                    break

            self.agent_manager.update_epsilon()

            if (episode + 1) % TARGET_UPDATE == 0:
                self.agent_manager.agent.target_net.load_state_dict(self.agent_manager.agent.policy_net.state_dict())
                print(f"Target network updated at episode {episode + 1}")

            avg_loss = episode_loss / optimization_step if optimization_step > 0 else 0
            avg_q = episode_q_value / optimization_step if optimization_step > 0 else 0

            episode_metrics = {
                'episode': episode,
                'reward': episode_reward,
                'length': step + 1,
                'avg_loss': avg_loss,
                'avg_q_value': avg_q,
                'epsilon': self.agent_manager.agent.epsilon
            }
            self.metrics_manager.save_metrics(episode_metrics)

            print(f"Episode: {episode}, Reward: {episode_reward}, Steps: {step + 1}, "
                  f"Avg Loss: {avg_loss:.4f}, Avg Q-Value: {avg_q:.4f}, Epsilon: {self.agent_manager.agent.epsilon:.2f}")

            if (episode + 1) % SAVE_MODEL_INTERVAL == 0:
                self.save_checkpoint(episode)

        self.save_checkpoint(NUM_EPISODES)
        self.env_manager.close()
        print("Training completed and environment closed.")
