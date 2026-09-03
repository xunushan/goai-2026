"""readback_cache.py — 子步级 GPU 读回缓存（优化 A 用，纯逻辑、无 Isaac 依赖）。

GPU 物理下每次 `.cpu()` 都是一次同步点（等待 GPU 排队工作 flush）。夹爪关节位姿
仅在物理子步推进时变化，而每次物理推进对应 DirectRLEnv._sim_step_counter 单调 +1，
故同一 (counter, 读回对象) 内的多次读取值必然相同 —— 只需整批读回一次，其余切片命中缓存。

正确性前提（由调用方保证）：
- epoch 必须随每次"可能改变被读数据的物理推进"严格递增（单调即可，不要求连续）；
- 同一 epoch 内不修改底层数据。

本类只负责"命中/淘汰"，不含任何 torch/Isaac 依赖，可独立单元测试。
"""


class EpochReadbackCache:
    def __init__(self):
        self._epoch = None
        self._rid = None
        self._data = None

    def get(self, epoch, rid):
        """命中返回缓存的 numpy 主数组；未命中/过期返回 None（数据永不为 None）。"""
        if self._epoch == epoch and self._rid == rid and self._data is not None:
            return self._data
        return None

    def put(self, epoch, rid, data):
        self._epoch = epoch
        self._rid = rid
        self._data = data

    def clear(self):
        self._epoch = None
        self._rid = None
        self._data = None
