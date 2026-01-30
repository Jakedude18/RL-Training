from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch

from wheelChairEnv import WheelChairEnv

# ==========================
# Custom callback for logging
# ==========================
class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        # Increment reward and length per step
        self.current_rewards += self.locals.get("rewards", 0)
        self.current_length += 1

        # Check if episode ended
        done_array = self.locals.get("dones", None)
        if done_array is not None and done_array.any():
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            if self.verbose > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_len = np.mean(self.episode_lengths[-10:])
                print(f"Episode finished. Avg reward (last 10): {avg_reward:.2f}, Avg length: {avg_len:.2f}")
            self.current_rewards = 0
            self.current_length = 0
        return True

# ==========================
# Wrap environment for SB3
# ==========================
env = DummyVecEnv([lambda: WheelChairEnv(simulation=False)])

# ==========================
# Create DQN model
# ==========================
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.1,
    buffer_size=10000,
    learning_starts=100,
    batch_size=64,
    tau=0.05,
    gamma=0.95,
    verbose=1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    policy_kwargs={"net_arch": [64, 64]},
)

# ==========================
# Train with callback
# ==========================
callback = TrainingMonitorCallback(verbose=1)
model.learn(total_timesteps=150, callback=callback)

# ==========================
# Evaluate policy on all states
# ==========================
def decode_action(action_idx):
    """Flat action 0-26 → [left, right, lift] in {-1,0,1}"""
    left = action_idx // 9
    right = (action_idx % 9) // 3
    lift = action_idx % 3
    return [left - 1, right - 1, lift - 1]

# Generate all possible states
all_states = []
for left in range(11):
    for right in range(11):
        for lift in range(11):
            all_states.append([left, right, lift])
all_states = np.array(all_states, dtype=np.float32)

# Get Q-values for all states
with torch.no_grad():
    q_values = model.q_net(torch.tensor(all_states))
    best_actions = torch.argmax(q_values, dim=1).numpy()

# Print best action per state
for state, action_idx in zip(all_states, best_actions):
    action = decode_action(action_idx)
    print(f"State {state.astype(int)} -> Best Action {[int(x) for x in action]}")



# After training
all_rewards = callback.episode_rewards
all_lengths = callback.episode_lengths

if all_lengths:
    avg_length = np.mean(all_lengths)
    avg_reward = np.mean(all_rewards)
    print(f"\nOverall average episode length: {avg_length:.2f} steps")
    print(f"Overall average episode reward: {avg_reward:.2f}")
else:
    print("No episodes completed during training.")