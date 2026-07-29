"""LeRobot 0.6 SmolVLA adapter for the RoboDojo policy server."""

from .deploy import eval_one_episode, eval_one_episode_batch
from .model import Model


def get_model(deploy_cfg):
    return Model(deploy_cfg)
