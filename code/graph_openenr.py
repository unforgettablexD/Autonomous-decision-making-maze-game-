import json
import matplotlib.pyplot as plt

# Path to your JSON file
json_file_path = 'agent_performance_data.json'

# Read the data from the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Assuming the JSON structure is like { "episodes": [...], "returns": [...] }
episodes = data["episodes"]
returns = data["returns"]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(episodes, returns, label='Agent Performance')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Agent Performance Over Episodes')
plt.legend()
plt.grid(True)
plt.show()
