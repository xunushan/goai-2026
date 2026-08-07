# ------------------------------------------------------------------------------
# xvla_2 policy-server 模型适配：直接复用 pip 安装的 xvla 包提供的
# RoboDojoPolicyClient（evaluation.robodojo）。只 import 已安装包，不 import
# XPolicyLab —— setup_policy_server.py 用 `Model` 鸭子类型实例化即可。
#
# setup_policy_server.py 按 `Model(deploy_cfg)` 构造；client 读取 deploy.yml /
# --overrides 中的 model/device/steps/domain_id/actions_per_chunk/log_io 等。
# ------------------------------------------------------------------------------
from evaluation.robodojo import RoboDojoPolicyClient

Model = RoboDojoPolicyClient
