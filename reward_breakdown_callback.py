from stable_baselines3.common.callbacks import BaseCallback

class RewardBreakdownCallback(BaseCallback):
    """
    Logs each reward component to TensorBoard using info dict from env.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        if len(infos) > 0 and 'reward_breakdown' in infos[0]:
            reward_breakdown = infos[0]['reward_breakdown']
            for key, value in reward_breakdown.items():
                self.logger.record(f'rewards/{key}', value)
        return True