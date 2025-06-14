from stable_baselines3.common.callbacks import BaseCallback

# Custom callback to log episode metrics to TensorBoard
class EpisodeMetricsLogger(BaseCallback):
    """
    Custom callback for logging episode-level metrics to TensorBoard during training.

    This callback accumulates rewards for each episode and logs the total reward and the number of resets
    whenever an episode ends. It is useful for monitoring training progress and debugging.

    Args:
        n_steps (int): Number of steps between logging events.
        verbose (int, optional): Verbosity level. If > 0, prints reward and reset info to stdout.

    Usage:
        episode_logger = EpisodeMetricsLogger(n_steps=2048, verbose=1)
    """
    def __init__(self, n_steps: int, verbose: int = 0):
        super(EpisodeMetricsLogger, self).__init__(verbose)
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