from gymnasium.envs.registration import register
from ..wheelChairEnv import WheelChairEnv

register(
    id="WheelChair-v0",
    entry_point="gymnasium_env.wheelChairEnv:WheelChairEnv",
    max_episode_steps=50,
)
