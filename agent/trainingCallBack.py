import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingMonitorCallback(BaseCallback):
    """
    SB3 callback for tracking episode rewards, lengths, and goal successes.
    Compatible with off-policy algorithms like DQN and SAC.
    Saves CSV and NPZ after training.
    """
    def __init__(self, save_path="episode_data", verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episodes_reached_goal = 0

        # Current episode buffers
        self.current_reward = 0.0
        self.current_length = 0

        # Buffers for full trajectory storage
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_next_states = []
        self.all_dones = []

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    def _on_step(self) -> bool:
        # --------------------------
        # Grab current step info
        # --------------------------
        obs = self.training_env.get_attr("state")[0]  # DummyVecEnv: env list
        obs = np.array(obs, dtype=np.int64)

        action = self.locals["actions"]
        if isinstance(action, np.ndarray):
            action = int(action[0])

        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        # --------------------------
        # Update current episode stats
        # --------------------------
        self.current_reward += reward
        self.current_length += 1

        # Append to trajectory buffers
        self.all_states.append(obs)
        self.all_actions.append(action)
        self.all_rewards.append(reward)
        self.all_next_states.append(obs)  # next_state placeholder
        self.all_dones.append(done)

        # Check if episode ended
        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)

            # Check for goal success
            if info.get("is_success", False):
                self.episodes_reached_goal += 1

            # Reset current episode stats
            self.current_reward = 0.0
            self.current_length = 0

        return True

    def _on_training_end(self) -> None:
        # --------------------------
        # Save CSV
        # --------------------------
        csv_file = self.save_path + ".csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["s0","s1","s2","action","reward","s0_next","s1_next","s2_next","done"]
            )
            for s, a, r, s_next, d in zip(
                self.all_states, self.all_actions, self.all_rewards, self.all_next_states, self.all_dones
            ):
                row = list(s) + [a, f"{r:.6f}"] + list(s_next) + [int(d)]
                writer.writerow(row)

        if self.verbose > 0:
            print(f"Saved episodes CSV to {csv_file}")
            print(f"The agent reached the goal {self.episodes_reached_goal} times")

        # --------------------------
        # Save NPZ
        # --------------------------
        npz_file = self.save_path + ".npz"
        np.savez(
            npz_file,
            states=np.array(self.all_states, dtype=np.int64),
            actions=np.array(self.all_actions, dtype=np.int64),
            rewards=np.array(self.all_rewards, dtype=np.float32),
            next_states=np.array(self.all_next_states, dtype=np.int64),
            dones=np.array(self.all_dones, dtype=np.int32),
        )
        if self.verbose > 0:
            print(f"Saved episode data NPZ to {npz_file}")
