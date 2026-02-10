import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wheelChairEnv import WheelChairEnv
import numpy as np

def objective(trial):
    # --------------------------
    # Suggest hyperparameters
    # --------------------------
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.1, 0.9)
    tau = trial.suggest_uniform("tau", 0.01, 0.1)
    max_steps = trial.suggest_int("max_steps", 50, 500)  # <--- tune this too


    # --------------------------
    # Create environment
    # --------------------------
    env = DummyVecEnv([lambda: WheelChairEnv(simulation=False, max_steps=200)])

    # --------------------------
    # Create DQN model
    # --------------------------
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        tau=tau,
        exploration_fraction=exploration_fraction,
        verbose=0,  # suppress training output
        policy_kwargs={"net_arch": [64, 64]},
    )

    # --------------------------
    # Train for a small number of timesteps
    # --------------------------
    model.learn(total_timesteps=1000)

    # --------------------------
    # Evaluate performance
    # --------------------------
    # Rollout one episode to get reward
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]  # DummyVecEnv wraps reward in list

    return total_reward  # higher is better


study = optuna.create_study(direction="maximize")  # we want max reward
study.optimize(objective, n_trials=1000)  # try 20 different hyperparameter sets

print("Best trial:")
trial = study.best_trial
print(trial.params)
