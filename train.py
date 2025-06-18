from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from callbacks.episode_metrics_logger import EpisodeMetricsLogger
from callbacks.reward_breakdown_callback import RewardBreakdownCallback
import configs as c
from nhl_hitz_env import NHLHitzGymEnv

from uuid import uuid4
import argparse


def train(session_id, train_pretrained, save_model, pretrained_model_path=None):
    """
    Train a PPO (Proximal Policy Optimization) model for NHL Hitz environment.

    This function handles the training process of a reinforcement learning model using Stable Baselines3's PPO implementation.
    It supports both training from scratch and continuing training from a pretrained model.

    Args:
        session_id (str): Unique identifier for the training session. Used for saving checkpoints and model files.
        train_pretrained (bool): If True, loads and continues training from a pretrained model identified by session_id.
                                If False, starts training from scratch.
        save_model (bool): If True, saves the final model after training.

    The function:
    - Creates and normalizes the NHL Hitz environment
    - Sets up callbacks for reward logging, checkpointing, and reward breakdown
    - Initializes or loads a PPO model with CNN policy
    - Trains the model for the specified number of steps
    - Saves the final model if save_model is enabled in configs
    """
    # get configs 
    cfg = c.basic

    train_steps = cfg["train_steps"]
    n_steps = cfg["n_steps"]
    verbose = cfg["verbose"]
    batch_size = cfg["batch_size"]
    n_epochs = cfg["n_epochs"]
    progress_bar = cfg["progress_bar"]

    # create environment
    env = DummyVecEnv([lambda: NHLHitzGymEnv(cfg)])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # initialize callbacks
    episode_metrics = EpisodeMetricsLogger(n_steps=n_steps, verbose=verbose)
    reward_breakdown_callback = RewardBreakdownCallback()

    checkpoint_freq = int((train_steps*n_steps) // 5)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix=session_id
    )

    callbacks = CallbackList([
        episode_metrics, 
        checkpoint_callback, 
        reward_breakdown_callback
    ])

    # create model
    model = PPO(
        'CnnPolicy', 
        env, 
        verbose=verbose,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_steps=n_steps,
        tensorboard_log="./tb_logs/"
    )
    
    # check to train on pretrained model
    reset_num_timesteps = not train_pretrained
    if train_pretrained:
        if pretrained_model_path:
            # Load model from specified path
            model_path = pretrained_model_path
        else:

            model_path = f"saved_models/{session_id}-model.zip"

        model = PPO.load(model_path, tensorboard_log="./tb_logs/")
        model.set_env(env)

        print(f"Successfully loaded model from {model_path}")

    model.learn(
        total_timesteps=train_steps*n_steps, 
        progress_bar=progress_bar,
        tb_log_name=session_id,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps
    )

    if save_model:
        model.save("saved_models/" + session_id + "-model")

    # close environments
    env.close()
    

if __name__ == "__main__":
    # create args parser
    parser = argparse.ArgumentParser(description='Train a reinforcement learning model')

    # set arguments
    parser.add_argument('--session_id', type=str, default=str(uuid4())[:5], help='session_id for the model')
    parser.add_argument('--train-pretrained', type=bool, default=False, help='Continue training a pretrained model. This will load the model from the session_id.')
    parser.add_argument('--save-model', type=bool, default=True, help='Save the model after training')
    parser.add_argument('--pretrained-model-path', type=str, default=None, help='Path to load pretrained model. Overrides session_id if provided.')

    # parse args
    args = parser.parse_args()

    # train model
    train(args.session_id, args.train_pretrained, args.save_model, args.pretrained_model_path)

