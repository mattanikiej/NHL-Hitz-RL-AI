from stable_baselines3 import PPO


from nhl_hitz_env import NHLHitzGymEnv
import configs as c
from stable_baselines3.common.vec_env import DummyVecEnv


if __name__ == "__main__":

    cfg = c.basic

    env = DummyVecEnv([lambda: NHLHitzGymEnv(cfg)])

    file_name = 'saved_models/68504-model.zip'
    model = PPO.load(file_name, env=env)
    model.verbose = cfg["verbose"]
    model.batch_size = cfg["batch_size"]


    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):  # increase this for the model to run longer
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)

    env.close()