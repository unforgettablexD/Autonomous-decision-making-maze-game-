import matplotlib.pyplot as plt
import numpy as np
import os
import json

import rooms  # Make sure rooms module is properly imported
from agent import (RandomAgent, SARSALearner, QLearner, TDLambdaLearner,
                                     PolicyGradientAgent, REINFORCEAgent, ActorCriticAgent,
                                     TDActorCriticAgent, TDLambdaActorCriticAgent)  # Import your agent classes

def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        action = agent.policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor ** time_step) * reward
        time_step += 1
    return discounted_return

def run_experiment(env, agent_class, params, training_episodes):
    agent = agent_class(params)
    returns = []
    for i in range(training_episodes):
        total_reward = episode(env, agent, i)
        returns.append(total_reward)
    return returns

def smooth_data(data, window_width):
    return np.convolve(data, np.ones(window_width)/window_width, mode='valid')

def main():
    data_filename = 'agent_performance_data_1.json'
    if os.path.exists(data_filename):
        with open(data_filename, 'r') as file:
            agents_returns = json.load(file)
    else:
        rooms_instance = "hard_0"
        env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
        params = {
            "nr_actions": env.action_space.n,
            "gamma": 0.99,
            "epsilon_decay": 0.0001,
            "alpha": 0.1,
            "env": env,
            "lambda": 0.8,  # For TDLambdaLearner and TDLambdaActorCriticAgent
        }
        training_episodes = 300

        agent_classes = [
            ('RandomAgent', RandomAgent),
            ('SARSALearner', SARSALearner),
            ('QLearner', QLearner),
            ('TDLambdaLearner', TDLambdaLearner),
            ('PolicyGradientAgent', PolicyGradientAgent),
            ('REINFORCEAgent', REINFORCEAgent),
            ('ActorCriticAgent', ActorCriticAgent),
            ('TDActorCriticAgent', TDActorCriticAgent),
            ('TDLambdaActorCriticAgent', TDLambdaActorCriticAgent)
        ]

        agents_returns = {}
        for agent_name, agent_class in agent_classes:
            print(f"Training {agent_name}...")
            agents_returns[agent_name] = run_experiment(env, agent_class, params, training_episodes)

        with open(data_filename, 'w') as file:
            json.dump(agents_returns, file)

    plt.figure(figsize=(10, 6))
    for agent_name, returns in agents_returns.items():
        smoothed_returns = smooth_data(returns, window_width=10)  # Adjust window width as needed
        plt.plot(range(len(smoothed_returns)), smoothed_returns, label=agent_name)

    plt.title('Smoothed Discounted Return Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Discounted Return (Smoothed)')
    plt.legend()
    plt.savefig('smoothed_discounted_return_comparison_1.png')
    plt.show()

if __name__ == "__main__":
    main()
