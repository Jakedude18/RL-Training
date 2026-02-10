from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import torch

from wheelChairEnv import WheelChairEnv
from trainingCallBack import TrainingMonitorCallback



env = DummyVecEnv([lambda: WheelChairEnv(simulation=False, max_steps=100)])

# ==========================
# Create DQN model
# ==========================
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.00021680265991082557,
    buffer_size=10000,
    learning_starts=100,
    batch_size=64,
    tau=0.046649157305473865,
    gamma=0.9072691785468209,
    verbose=1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.4817159675707373,  # decay over fraction% of timesteps
    policy_kwargs={"net_arch": [64, 64]},
)

# ==========================
# Train with callback
# ==========================
callback = TrainingMonitorCallback(verbose=1)
model.learn(total_timesteps=1000, callback=callback)

# ==========================
# Evaluate policy on all states
# ==========================

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


def decode_action(action_idx):
    """Flat action 0-26 → [left, right, lift] in {-1,0,1}"""
    left = action_idx // 9
    right = (action_idx % 9) // 3
    lift = action_idx % 3
    return [left - 1, right - 1, lift - 1]


# Print best action per state
for state, action_idx in zip(all_states, best_actions):
    action = decode_action(action_idx)
    # print(f"State {state.astype(int)} -> Best Action {[int(x) for x in action]}")



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