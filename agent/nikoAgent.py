from collections import defaultdict
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET = [10, 10]

class nikoAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        n_actions = np.prod(self.env.action_space.nvec)  # total number of discrete actions
        self.q_values = defaultdict(lambda: np.ones(n_actions)*0.1)

        self.lr = learning_rate

        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    @classmethod
    def decode_action(cls, a: int) -> np.ndarray:
        """Convert action index → actuator commands (-1,0,1)"""
        a = int(a)
        #convert 1-27 to base 3
        base3 = np.base_repr(a, base=3).zfill(3)
        #convert from 0,1,2 to -1,0,1
        return np.array([int(d) - 1 for d in base3])

    @classmethod
    def encode_action(cls, acts: np.ndarray) -> int:
        """Convert actuator commands (-1,0,1) → action index"""
        #add 1 to each int in the array and turn them into string array
        digits = [str(a + 1) for a in acts]
        #Interpret string as base 3 number!
        return int("".join(digits), 3)
    
    def get_action(self, obs: list[int]) -> int:
        """Choose an action using epsilon-greedy strategy."""

        n_actions = np.prod(self.env.action_space.nvec)  # total number of discrete actions

        if np.random.random() < self.epsilon:
            return np.random.randint(n_actions)

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[tuple(obs)]))

   
    def update(self, obs: np.ndarray, action: int, reward: float, terminated: bool, next_obs: np.ndarray):
        obs_key = tuple(obs)
        next_key = tuple(next_obs)

        # future Q-value 0 if episode ended
        future_q_value = 0 if terminated else np.max(self.q_values[next_key])

        # TD target
        target = reward + self.discount_factor * future_q_value

        # TD error
        td_error = target - self.q_values[obs_key][action]

        # Q-value update
        self.q_values[obs_key][action] += self.lr * td_error

        # Optionally clip Q-values to avoid negative numbers
        # self.q_values[obs_key][action] = max(0, self.q_values[obs_key][action])

        # Store TD error
        self.training_error.append(td_error)


    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def display_q_values(self):
        """Print only the best action per state with its Q-value."""
        data = []
        for state, values in self.q_values.items():
            best_idx = int(np.argmax(values))
            best_action = nikoAgent.decode_action(best_idx)
            best_value = values[best_idx]
            data.append([*state[:3], best_action, best_value])

        df = pd.DataFrame(data, columns=["Left Panel", "Right Panel", "Main Panel", "Best Action", "Q-value"])
        df = df.sort_values(by=["Left Panel", "Right Panel", "Main Panel"]).reset_index(drop=True)
        print(df.to_string(index=False))



    def plot_policy(agent, lift_level=0):
        xs, ys = [], []
        us, vs = [], []

        for left in range(11):
            for right in range(11):
                state = (left, right, lift_level)
                dx, dy = get_policy_direction(agent, state)

                xs.append(left)
                ys.append(right)
                us.append(dx)
                vs.append(dy)

        plt.figure(figsize=(7, 7))
        plt.quiver(xs, ys, us, vs, angles='xy', scale_units='xy', scale=1)

        # Target marker
        plt.scatter([TARGET[0]], [TARGET[1]], marker='*', s=200)

        plt.xlim(-0.5, 10.5)
        plt.ylim(-0.5, 10.5)
        plt.xticks(range(11))
        plt.yticks(range(11))
        plt.grid(True)

        plt.xlabel("Left Paddle")
        plt.ylabel("Right Paddle")
        plt.title(f"Learned Policy (lift={lift_level})")

        plt.show()



def get_policy_direction(agent, state):
    """
    Returns (dx, dy) for the greedy action at this state.
    """
    q_vals = agent.q_values.get(state)
    if q_vals is None:
        return 0, 0  # unseen state

    best_idx = int(np.argmax(q_vals))
    action = agent.decode_action(best_idx)

    dx = action[0]  # left paddle change
    dy = action[1]  # right paddle change
    return dx, dy
