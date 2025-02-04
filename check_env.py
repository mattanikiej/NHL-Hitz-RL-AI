from stable_baselines3.common.env_checker import check_env
import configs as c
from nhl_hitz_env import NHLHitzGymEnv


cfg = c.basic

# create environment
env = NHLHitzGymEnv(cfg)

# check environment
check_env(env)

# close environments
env.close()
