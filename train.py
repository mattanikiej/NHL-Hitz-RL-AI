from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import configs as c
from nhl_hitz_env import NHLHitzGymEnv


cfg = c.basic
train_steps = cfg["train_steps"]
n_steps = cfg["n_steps"]

# create environment
env = NHLHitzGymEnv(cfg)

# uncomment to check environment if you made any changes
# check_env(env)

model = PPO('CnnPolicy', 
            env, 
            verbose=1,
            batch_size=128,
            n_epochs=1,
            n_steps=n_steps)

model.learn(total_timesteps=train_steps*n_steps, progress_bar=True)

# close environments
env.close()
