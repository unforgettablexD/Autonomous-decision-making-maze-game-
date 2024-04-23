import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import json
import pickle 


def episode(env, agent,pos = None,  nr_episode=0):
    state = env.reset()

    #The following change was added to the code so that we could test our agent with custom positions.
    if pos: 
        env.agent_position = pos

    discounted_return = 0
    discount_factor = 0.997
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
params["epsilon_decay"] = 0.000025
params["alpha"] = 0.2
params["env"] = env

#agent = a.RandomAgent(params)
#agent = a.SARSALearner(params)
#agent = a.QLearner(params)
params['lambda'] = 0.8  # Or another value you wish to use
agent = a.TDLambdaLearner(params)

training_episodes = 2000
returns = [episode(env, agent,None, i) for i in range(training_episodes)]
x = range(training_episodes)
y = returns

#Save the agent so that he (she? / they/them?) can be reused
# with open('agent.pkl' , 'wb') as file: 
#     pickle.dump(agent , file)


plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.savefig("agent1_progress.png")
plot.show()

# env.save_video()
