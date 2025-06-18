from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nhl_hitz_env import NHLHitzGymEnv
import configs as c

import argparse


def run_pretrained_model(model_path, steps):
    """
    Run a pretrained PPO model on the NHL Hitz environment.
    
    This function loads a previously trained PPO model and runs it on the NHL Hitz
    gym environment for a specified number of steps. The environment is set up with
    the basic configuration and reward normalization.
    
    Args:
        model_path (str): Path to the saved PPO model file (.zip format)
        steps (int): Number of steps to run the model. Use -1 for infinite steps.
    
    Returns:
        None
        
    Note:
        The function automatically closes the environment when finished.
        The model runs in deterministic mode for consistent behavior.
    """
    cfg = c.basic

    env = DummyVecEnv([lambda: NHLHitzGymEnv(cfg)])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    model = PPO.load(model_path, env=env)
    model.batch_size = cfg["batch_size"]

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()

    if steps == -1:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
    else:
        for i in range(steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)


    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runs pretrained model")

    parser.add_argument('--model-path', type=str, default='saved_models/hawksai-model', help="Path to the saved model")
    parser.add_argument('--steps', type=int, default=500, help="Number of steps for the model to take, -1 for infinite")

    args = parser.parse_args()

    run_pretrained_model(args.model_path, args.steps)