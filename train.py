from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_logging_callback import RewardLoggingCallback
from reward_breakdown_callback import RewardBreakdownCallback
import configs as c
from nhl_hitz_env import NHLHitzGymEnv

from uuid import uuid4
import argparse


def train(session_id):

    # get configs 
    cfg = c.basic

    train_steps = cfg["train_steps"]
    n_steps = cfg["n_steps"]
    verbose = cfg["verbose"]
    batch_size = cfg["batch_size"]
    n_epochs = cfg["n_epochs"]
    progress_bar = cfg["progress_bar"]
    save_model = cfg["save_model"]

    # create environment
    env = DummyVecEnv([lambda: NHLHitzGymEnv(cfg)])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # initialize callbacks
    reward_callback = RewardLoggingCallback(n_steps=n_steps, verbose=verbose)

    reward_breakdown_callback = RewardBreakdownCallback()

    checkpoint_freq = int((train_steps*n_steps) // 5)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix=session_id,
    )

    callbacks = CallbackList([
        reward_callback, 
        checkpoint_callback, 
        reward_breakdown_callback
        ])

    # create model
    model = PPO('CnnPolicy', 
                env, 
                verbose=verbose,
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
                tensorboard_log="./tb_logs/")
    
    # check to train on pretrained model
    reset_num_timesteps = True
    if cfg["train_pretrained"]:
        session_id = cfg["session_id"]

        model_path = f"saved_models/{session_id}-model.zip"

        model = PPO.load(model_path, tensorboard_log="./tb_logs/")
        model.set_env(env)
        
        reset_num_timesteps = False

    model.learn(total_timesteps=train_steps*n_steps, 
                progress_bar=progress_bar,
                tb_log_name=session_id,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps)

    if save_model:
        model.save("saved_models/" + session_id + "-model")

    # close environments
    env.close()
    

if __name__ == "__main__":
    # create args parser
    parser = argparse.ArgumentParser(description='Train a reinforcement learning model')

    # set arguments
    parser.add_argument('--session_id', type=str, default=str(uuid4())[:5], help='session_id for the model')

    # parse args
    args = parser.parse_args()

    # train model
    train(args.session_id)

