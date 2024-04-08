import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import agent as a
import rooms
import os
import json
import rooms  # Assuming you have a module called 'rooms' that includes the environment

def episode(env, agent, nr_episode=0):
    state = env.reset()
    # print(env.obstacles)
    # return
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor ** time_step) * reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
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
    # Check if the data file exists to avoid re-running the experiments
    data_filename = 'agent_performance_data_medium1.json'
    if os.path.exists(data_filename):
        with open(data_filename, 'r') as file:
            agents_returns = json.load(file)
    else:
        # Initialize the environment
        rooms_instance = "hard_0"
        env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")

        # Initialize common parameters, including 'env'
        params = {
            "nr_actions": env.action_space.n,
            "gamma": 0.99,
            "epsilon_decay": 0.0001,
            "alpha": 0.1,
            "env": env,
            "lambda": 0.8,  # For TDLambdaLearner
        }

        # Define the number of episodes for training
        training_episodes = 600

        # List of agent classes to compare
        agent_classes = [
            ('RandomAgent', a.RandomAgent),
            ('SARSALearner', a.SARSALearner),
            ('QLearner', a.QLearner),
            ('TDLambdaLearner', a.TDLambdaLearner),
        ]

        # Run experiment for each agent and collect returns
        agents_returns = {}
        for agent_name, agent_class in agent_classes:
            print(f"Training {agent_name}...")
            agents_returns[agent_name] = run_experiment(env, agent_class, params, training_episodes)

        # Save the raw data to a file for future use
        with open(data_filename, 'w') as file:
            json.dump(agents_returns, file)

    # Plotting the performance of each agent with smoothing
    plt.figure(figsize=(10, 6))
    for agent_name, returns in agents_returns.items():
        smoothed_returns = smooth_data(returns, window_width=10)  # Change window_width as needed
        plt.plot(range(len(smoothed_returns)), smoothed_returns, label=agent_name)

    plt.title('Smoothed Agent Performance Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Smoothed)')
    plt.legend()
    plt.savefig('smoothed_agent_performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()