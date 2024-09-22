# scripts/evaluate.py

import argparse
from evaluators.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained DDQN Agent vs Random Agent")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained agent checkpoint')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--save', type=str, default='evaluation_results.json', help='Path to save evaluation results')
    args = parser.parse_args()

    evaluator = Evaluator(agent_checkpoint_path=args.checkpoint, num_episodes=args.episodes, render=args.render)
    
    print("Evaluating Trained Agent...")
    trained_rewards, trained_steps = evaluator.evaluate_agent(render=args.render)
    
    print("\nEvaluating Random Agent...")
    random_rewards, random_steps = evaluator.evaluate_random_agent(render=args.render)
    
    evaluator.visualize_results(trained_rewards, random_rewards)
    
    evaluator.save_results(trained_rewards, random_rewards, save_path=args.save)
    
    evaluator.close()

if __name__ == "__main__":
    main()