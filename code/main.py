import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import json

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
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return
    
params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["env"] = env

#agent = a.RandomAgent(params)
#agent = a.SARSALearner(params)
#agent = a.QLearner(params)
params['lambda'] = 0.8  # Or another value you wish to use
agent = a.TDLambdaLearner(params)

#agent = a.TDLambdaLearner(params)
training_episodes = 200
returns = [episode(env, agent, i) for i in range(training_episodes)]

graph_data = {
    "episodes": list(range(training_episodes)),
    "returns": returns
}
with open('agent_tdlambda_graph4.json', 'w') as json_file:
    json.dump(graph_data, json_file)

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.savefig("agent1_progress.png")
plot.show()

env.save_video()
