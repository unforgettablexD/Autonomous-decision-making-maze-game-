import json
import matplotlib.pyplot as plt
import numpy as np
import os

# List of JSON files containing the graph data for each agent
json_files = ['agent_random_graph.json', 'agent_qlearner_graph.json', 'agent_sarsa_graph.json', 'agent_tdlambda_graph.json']
# Corresponding names for the agents to label the plots
agent_names = ['Random Agent', 'SARSA Learner', 'Q-Learner', 'TD Lambda Learner']


def smooth_data(data, window_width=5):
    """Applies a moving average to smooth the data."""
    return np.convolve(data, np.ones(window_width) / window_width, mode='valid')


def plot_graphs(json_files, agent_names):
    plt.figure(figsize=(10, 6))

    # Ensure the same number of agent names and JSON files are provided
    assert len(json_files) == len(agent_names), "Each JSON file must have a corresponding agent name."

    for file_name, agent_name in zip(json_files, agent_names):
        # Check if the JSON file exists
        if not os.path.exists(file_name):
            print(f"File {file_name} not found. Skipping.")
            continue

        # Read data from the JSON file
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)

        # Smooth the returns data
        smoothed_returns = smooth_data(data['returns'])

        # Plot the smoothed returns data
        plt.plot(data['episodes'][:len(smoothed_returns)], smoothed_returns, label=agent_name)

    # Add titles and labels
    plt.title('Combined Agent Performance Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Smoothed)')
    plt.legend()

    # Save the combined plot
    plt.savefig('combined_agent_performance.png')

    # Display the plot
    plt.show()


# Call the function to plot the graphs
plot_graphs(json_files, agent_names)
