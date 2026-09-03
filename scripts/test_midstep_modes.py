#!/usr/bin/env python3
"""本地逻辑单测：X_VLA deploy.py 中间步三档 full|video|none（无需 GPU/Isaac）。

用法：cd goai_2026 && python3 scripts/test_midstep_modes.py

覆盖的不变式：
1) full（默认）：中段每步 get_obs_batch(全量) + update_obs_batch 推服务端（历史行为）。
2) video：中段只调 get_obs_batch(vision_only=True)，不 update_obs_batch；while 顶部
   全量 get_obs_batch + update_obs_batch 仍照常（真 obs 路径不变）。
3) none：中段不调 get_obs_batch/update_obs_batch；while 顶部照常。
4) 三档 while 顶部行为一致（每 chunk 一次全量 obs + 一次 get_action）。
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_PATH = os.path.join(ROOT, "RoboDojo/XPolicyLab/policy/X_VLA/deploy.py")


def _load():
    spec = importlib.util.spec_from_file_location("xvla_deploy_mod", DEPLOY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEPLOY = _load()

CHUNK = 3  # 每 chunk 动作数（用 3 以保留中段步）
STEPS_TILL_END = CHUNK  # 一个 chunk 后 episode 结束


class FakeEnv:
    def __init__(self):
        self.n_take = 0
        self.full_obs = 0      # get_obs_batch(vision_only=False)
        self.video_obs = 0     # get_obs_batch(vision_only=True)
        self.log = []

    def get_running_env_idx_list(self):
        # episode 结束后无 running env（while 退出）
        return [] if self.n_take >= STEPS_TILL_END else [0]

    def get_obs_batch(self, env_idx_list, vision_only=False, last_frame=False):
        key = "video" if vision_only else "full"
        if key == "video":
            self.video_obs += 1
        else:
            self.full_obs += 1
        self.log.append(("obs", key))
        return [{"ep": 0}] if not vision_only else []

    def take_action_batch(self, current_action_list, env_idx_list):
        self.n_take += 1
        self.log.append(("take", env_idx_list[:]))

    def is_episode_end(self):
        return self.n_take >= STEPS_TILL_END


class FakeClient:
    def __init__(self):
        self.calls = []
        self.reset_seen = 0

    def call(self, func_name, **kw):
        self.calls.append((func_name, kw))
        if func_name == "reset":
            self.reset_seen += 1
            return None
        if func_name == "update_obs_batch":
            return None
        if func_name == "get_action_batch":
            return [list(range(CHUNK))]  # 单 env 的 chunk
        return None


def run(mode):
    env = FakeEnv()
    client = FakeClient()
    DEPLOY._mid_step_mode = lambda: mode
    DEPLOY.eval_one_episode_batch(env, client)
    # 过滤掉 reset / while 顶部的记录，只按计数断言
    return env, client


def main():
    for mode in ("full", "video", "none"):
        env, client = run(mode)
        # while 顶部：每 chunk 一次全量 obs + update + get_action
        upd = [c for c in client.calls if c[0] == "update_obs_batch"]
        ga = [c for c in client.calls if c[0] == "get_action_batch"]
        assert len(ga) == 1, (mode, "while 顶部应恰 1 次 get_action", len(ga))
        if mode == "full":
            assert env.video_obs == 0 and env.full_obs == 1 + (CHUNK - 1), (
                mode, "full 中段应全量 obs 每步", env.video_obs, env.full_obs)
            assert len(upd) == 1 + (CHUNK - 1), (mode, "full 中段应每步 update", len(upd))
        elif mode == "video":
            assert env.full_obs == 1, (mode, "video 顶部 1 次全量 obs", env.full_obs)
            assert env.video_obs == CHUNK - 1, (mode, "video 中段应每步 video-only obs", env.video_obs)
            assert len(upd) == 1, (mode, "video 中段不应 update_obs_batch", len(upd))
        elif mode == "none":
            assert env.video_obs == 0 and env.full_obs == 1, (mode, "none 中段应无任何 obs", env.video_obs, env.full_obs)
            assert len(upd) == 1, (mode, "none 中段不应 update_obs_batch", len(upd))
        print(f"PASS mode={mode}: top full_obs=1/update=1/get_action=1; "
              f"mid full_obs={env.full_obs - 1} video_obs={env.video_obs} update_mid={len(upd) - 1}")
    # 非法值回退 full
    env, client = run("bogus")
    assert env.video_obs == 0 and env.full_obs == CHUNK, ("bogus 应回退 full")
    print("PASS mode=bogus: 回退 full")
    print("\nALL MIDSTEP MODE TESTS PASSED")


if __name__ == "__main__":
    main()
