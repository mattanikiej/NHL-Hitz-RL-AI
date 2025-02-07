from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import configs as c
from nhl_hitz_env import NHLHitzGymEnv
from uuid import uuid4


# Custom callback to log rewards to TensorBoard
class RewardLoggingCallback(BaseCallback):
    def __init__(self, n_steps: int, verbose: int = 0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.n_steps = n_steps
        self.episode_rewards = 0
        self.resets = 0

    def _on_step(self) -> bool:
        # Collect the reward for the current step
        reward = self.locals['rewards'][0]
        self.episode_rewards += reward

        # Log rewards to TensorBoard after every n_steps
        if self.locals['dones'][0]:
            self.logger.record('total_reward', self.episode_rewards)
            self.logger.record('resets', self.resets)

            if self.verbose > 0:
                print(f"Reset: {self.resets}, Reward: {self.episode_rewards:.2f}")

            # Reset the episode rewards
            self.episode_rewards = 0
            self.resets += 1

        return True 
    

if __name__ == "__main__":

    # create session id
    uuid = str(uuid4())[:5]

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

    # initialize callbacks
    reward_callback = RewardLoggingCallback(n_steps=n_steps, verbose=verbose)

    checkpoint_callback = CheckpointCallback(
        save_freq=(train_steps*n_steps) // 5,
        save_path="./checkpoints/",
        name_prefix=uuid,
    )

    callback = CallbackList([reward_callback, checkpoint_callback])

    model = PPO('CnnPolicy', 
                env, 
                verbose=verbose,
                batch_size=batch_size,
                n_epochs=n_epochs,
                n_steps=n_steps,
                tensorboard_log="./tb_logs/")

    model.learn(total_timesteps=train_steps*n_steps, 
                progress_bar=progress_bar,
                tb_log_name=uuid,
                callback=reward_callback)

    if save_model:
        model.save("saved_models/" + uuid + "-model")

    # close environments
    env.close()
