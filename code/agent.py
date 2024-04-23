import random
import numpy as np
from multi_armed_bandits import *
from collections import deque
from rooms import *


MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3

ROOMS_ACTIONS = [MOVE_NORTH,MOVE_SOUTH,MOVE_WEST,MOVE_EAST]

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, terminated, truncated):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent base for learning Q-values.
"""
class TemporalDifferenceLearningAgent(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon = 1.0
        
    def Q(self, state):
        state = np.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        Q_values = self.Q(state)
        return epsilon_greedy(Q_values, None, epsilon=self.epsilon)
    
    def decay_exploration(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_decay)

"""
 Autonomous agent using on-policy SARSA.
"""
class SARSALearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            next_action = self.policy(next_state)
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error
        
"""
 Autonomous agent using off-policy Q-Learning.
"""
class QLearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error


class TDLambdaLearner(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.lambda_ = params.get('lambda', 0.8)  # Default λ value is 0.8
        # Initialize eligibility traces
        self.eligibility_traces = {}
        self.initialize_q_values(params['env'])
        self.env = params['env']

    def test_mode(self):
        """
        Sets epsilon to 0 for testing
        """
        self.epsilon = 0 

    def Q(self, state, action=None):
        """
        Returns the Q value(s) for given state and action.
        If action is None, returns all action Q values for the given state.

        Parameters:
        - state: The current state, can be a numpy array or a string representation of the state.
        - action: The action for which Q value is queried. If None, all Q values for the state are returned.

        Returns:
        - A single Q value if action is specified, or a dictionary of action: Q value pairs if action is None.
        """
        # Check if the state is already a string, if not, convert it
        if isinstance(state, str):
            state_str = state
        else:
            # Ensure array2string is used only on appropriate data types
            try:
                state_str = np.array2string(state)
            except AttributeError:
                # Handle cases where conversion isn't straightforward
                # You might need to log this or handle it based on your application's requirements
                print("Error converting state to string:", state)
                return 0 if action is not None else {}

        # If action is specified, return the Q value for the specific state-action pair
        if action is not None:
            return self.Q_values.get((state_str, action), 0.0)

        # If action is None, construct a dictionary of all Q values for the current state
        all_actions_q_values = {a: self.Q_values.get((state_str, a), 0.0) for a in range(self.nr_actions)}
        return all_actions_q_values

    def decay_eligibility_traces(self):
        """
        Decays all eligibility traces by multiplying them with gamma * lambda.
        This method is called after each step in the episode.
        """
        for key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[key] *= self.gamma * self.lambda_

    def update(self, state, action, reward, next_state, terminated, truncated):
        """
        Updates the agent's knowledge based on the received experience tuple (state, action, reward, next_state)
        and the termination status of the episode. This function uses temporal difference (TD) learning
        combined with eligibility traces to update Q-values and adjusts exploration behavior.

        Parameters:
        - state (np.ndarray): The current state of the environment.
        - action (int): The action taken in the current state.
        - reward (float): The reward received after taking the action.
        - next_state (np.ndarray): The state of the environment after the action is taken.
        - terminated (bool): A flag indicating if the episode has ended due to the agent reaching a terminal state.
        - truncated (bool): A flag indicating if the episode has ended due to reaching a time limit.
        """

        # Decay exploration 
        self.decay_exploration()

        # Convert state to string for consistency
        state_str = np.array2string(state)
        next_state_str = np.array2string(next_state)

        # Update eligibility trace for the taken action
        self.eligibility_traces[(state_str, action)] = self.eligibility_traces.get((state_str, action), 0) + 1

        self.decay_eligibility_traces()

        # Determine best next action's Q-value for updating (in Q-Learning style)
        max_next_q = max(self.Q(next_state).values())

        # TD target with bootstrapping from next state
        td_target = reward + self.gamma * max_next_q * (not terminated)

        # Update Q-values and eligibility traces for all state-action pairs
        for (s_a, trace_value) in self.eligibility_traces.items():
            s, a = s_a
            q_val = self.Q(s, a)
            td_error = td_target - q_val
            self.Q_values[(s, a)] = q_val + self.alpha * td_error * trace_value

            # Decay the eligibility trace
            self.eligibility_traces[s_a] *= self.gamma * self.lambda_

        # Handle terminal state by resetting traces
        if terminated or truncated:
            self.reset_eligibility_traces()

    def reset_eligibility_traces(self):
        """
        Resets the eligibilty trace dictionary 
        """
        self.eligibility_traces = {}

    def decay_exploration(self):
        """
        Decays epsilon value to reduce exploration 
        """
        # Decay the epsilon value for ε-greedy policy
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.001)

    def policy(self, state):
        """
        Returns action based on the learnt policy. The policy also uses epsilon greedy to balance exploration and exploitation
        
        Parameters:
        - state (np.ndarray): The current state of the environment.
        
        """
        # Select action based on epsilon-greedy policy
        q_values = self.Q(state)
        action_values = [q_values[a] for a in range(self.nr_actions)]
        return epsilon_greedy(action_values, None, epsilon=self.epsilon)

    def initialize_q_values(self, env):
        """
        Initializes the Q-values for all state-action pairs based on a heuristic derived from the
        Breadth-First Search (BFS) distance to the goal state, adjusted for obstacles.

        This method computes the heuristic Q-value for each state by considering the shortest path
        to the goal that avoids obstacles. The heuristic is inversely proportional to the BFS distance,
        so that states closer to the goal have higher heuristic values.

        Parameters:
        - env (gym.Env): The environment instance which provides the grid dimensions,
        the locations of obstacles, and the goal position.
      """

        max_distance = env.width + env.height  # Used for fallback heuristic calculation
        goal_x, goal_y = env.goal_position
        obstacle_value = -1  # Value for a wall or obstacle
        obstacles = set(env.obstacles)

        def bfs_distance(start):
            """Returns the BFS distance to the goal from start, considering obstacles."""
            if start == (goal_x, goal_y):
                return 0
            visited = set([start])
            queue = deque([(start, 0)])  # Each item is (position, distance)
            while queue:
                (x, y), dist = queue.popleft()
                for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:  # 4 directions: N, E, S, W
                    next_pos = (x + dx, y + dy)
                    if next_pos not in visited and next_pos not in obstacles:
                        if next_pos == (goal_x, goal_y):
                            return dist + 1
                        if self.is_within_bounds(next_pos, env):
                            visited.add(next_pos)
                            queue.append((next_pos, dist + 1))
            return max_distance  # Fallback if no path is found

        for x in range(env.width):
            for y in range(env.height):
                state = np.array([x, y])
                state_str = np.array2string(state)
                distance = bfs_distance((x, y))

                # Calculate heuristic value based on BFS distance
                heuristic_value = (max_distance - distance) / max_distance                

                for action in range(env.action_space.n):
                    next_pos = self.predict_next_position(x, y, action, env)
                    if next_pos in env.obstacles or not self.is_within_bounds(next_pos, env):
                        # Set Q-value to -1 if the ensuing state is an obstacle
                        self.Q_values[(state_str, action)] = obstacle_value
                    else:
                        # Set Q-value based on BFS distance to the goal and strategic positioning
                        self.Q_values[(state_str, action)] = heuristic_value



    def predict_next_position(self, x, y, action, env):
        """
        Predicts the next position of the agent given its current state and action

        Parameters: 
        - x : Current x coordinate of the agent 
        - y : Current y coordinate of the agent
        - action : Action to be taken
        """
        if action == MOVE_NORTH:
            return (x, y - 1)
        elif action == MOVE_SOUTH:
            return (x, y + 1)
        elif action == MOVE_WEST:
            return (x - 1, y)
        elif action == MOVE_EAST:
            return (x + 1, y)
        return (x, y)  # Return the same position if action is unrecognized

    def is_within_bounds(self, position, env):
        """
        Determines if the current position of the agent is within bounds of a given environment

        Parameter: 
        - position : an (x,y) tuple that denotes an agents position in the environment
        - env : Environment under consideration 
        """
        x, y = position
        return 0 <= x < env.width and 0 <= y < env.height
