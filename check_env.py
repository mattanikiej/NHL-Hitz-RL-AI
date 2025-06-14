from stable_baselines3.common.env_checker import check_env
import configs as c
from nhl_hitz_env import NHLHitzGymEnv


cfg = c.basic

# create environment
print(f"Creating environment with basic config...")
env = NHLHitzGymEnv(cfg)

print(f"Successfully created environment {type(env)}")

# check environment
print(f"Checking environemnt {type(env)}...")
check_env(env)

print(f"Successfully checked environment {type(env)}")

# close environments
print(f"Closing environment {type(env)}...")
env.close()

print(f"Successfully closed environment {type(env)}")
print(f"No isses were found. You may train on environment {type(env)}")
