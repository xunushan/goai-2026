"""eval_one_episode_batch（deploy.py）的逐 env 动作缓冲回归测试。

无 Isaac / 无 torch，纯桩件。运行：python3 test_deploy_batch.py

不同 env 可能收到不同长度的 chunk，批评估不能按「批内统一 chunk 长度」推进，
因此每个 env 维护自己的动作缓冲。这里验证：

1. 各 env chunk 等长时，新实现与旧的按批统一长度实现在**动作序列**
   与**推理调用序列**上完全一致 —— 保证不动已入库的基线口径；
2. 各 env chunk 不等长时，每个 env 恰好在自己的缓冲耗尽后重规划；
3. 各 env 在不同步结束时不串扰、不残留缓冲；
4. 每步都取全量 obs（render/capture/视频写盘依赖它）。
"""
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_xvla_deploy_under_test", _HERE / "deploy.py")
_deploy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_deploy)
eval_one_episode_batch = _deploy.eval_one_episode_batch


# ---------------------------------------------------------------------------
# 桩件
# ---------------------------------------------------------------------------

class FakeEnv:
    """模拟 TASK_ENV 的批接口。episode 长度由 lengths 给出，逐步自增。"""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.num_envs = len(lengths)
        self.step_cnt = [0] * self.num_envs
        self.end_flag = [False] * self.num_envs
        self.actions = []          # (env_idx, step, action)
        self.obs_calls = []        # 每次 get_obs_batch 的 env 子集
        self.end_calls = 0

    def is_episode_end(self):
        self.end_calls += 1
        for env_idx in range(self.num_envs):
            if not self.end_flag[env_idx] and self.step_cnt[env_idx] >= self.lengths[env_idx]:
                self.end_flag[env_idx] = True
        return all(self.end_flag)

    def get_running_env_idx_list(self):
        return [i for i in range(self.num_envs) if not self.end_flag[i]]

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        env_idx_list = list(env_idx_list)
        self.obs_calls.append(tuple(env_idx_list))
        return [{"env_idx": i} for i in env_idx_list]

    def take_action_batch(self, actions_list, env_idx_list=None):
        for action, env_idx in zip(actions_list, env_idx_list):
            if self.end_flag[env_idx]:
                continue
            self.actions.append((env_idx, self.step_cnt[env_idx], action))
            self.step_cnt[env_idx] += 1


class FakeModelClient:
    """chunk_len(env_idx, query_no) -> 该次推理返回的 chunk 长度。

    忠实模拟 client_server/ws/model_client.py 的契约：update_obs_batch 只在本地存下
    观测列表，get_action_batch **忽略自己的 obs 参数**，按存下的那份逐个 env 推理。
    """

    def __init__(self, chunk_len):
        self.chunk_len = chunk_len
        self.calls = []            # (func_name, env 子集)
        self.n_queries = {}
        self._latest_obs_batch = None

    def call(self, func_name=None, obs=None, **kwargs):
        if func_name == "reset":
            self.calls.append(("reset", ()))
            self.n_queries = {}
            self._latest_obs_batch = None
            return None
        if func_name == "update_obs_batch":
            self._latest_obs_batch = list(obs)
            self.calls.append(("update_obs_batch", tuple(o["env_idx"] for o in obs)))
            return None
        if func_name == "get_action_batch":
            assert self._latest_obs_batch is not None, "get_action_batch 前必须 update_obs_batch"
            envs = tuple(o["env_idx"] for o in self._latest_obs_batch)
            self.calls.append(("get_action_batch", envs))
            out = []
            for env_idx in envs:
                q = self.n_queries.get(env_idx, 0)
                self.n_queries[env_idx] = q + 1
                out.append([f"e{env_idx}q{q}s{i}" for i in range(self.chunk_len(env_idx, q))])
            return out
        raise AssertionError(f"unexpected call {func_name}")

    def inference_envs(self):
        return [envs for name, envs in self.calls if name == "get_action_batch"]


def _old_eval_one_episode_batch(TASK_ENV, model_client):
    """改动前的按批内统一 chunk 长度推进实现，作为等长 chunk 的参照。"""
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_equal_chunks_match_old_implementation():
    """各 env chunk 等长时，新旧实现动作序列一致。"""
    lengths = [50, 70, 40]
    old_env, old_mc = FakeEnv(lengths), FakeModelClient(lambda e, q: 30)
    new_env, new_mc = FakeEnv(lengths), FakeModelClient(lambda e, q: 30)

    _old_eval_one_episode_batch(old_env, old_mc)
    eval_one_episode_batch(new_env, new_mc)

    assert old_env.actions == new_env.actions, "动作序列必须逐条一致"
    assert old_mc.inference_envs() == new_mc.inference_envs(), (
        old_mc.inference_envs(), new_mc.inference_envs())
    # 3 个 env，长度 50/70/40，chunk 30 → 各 env 各查询 ceil(len/30) 次
    assert new_mc.inference_envs()[0] == (0, 1, 2)
    assert {e: sum(e in envs for envs in new_mc.inference_envs()) for e in range(3)} == {0: 2, 1: 3, 2: 2}


def test_variable_chunks_replan_at_each_horizon():
    """长度为 9 和 30 的 chunk 分别耗尽后重规划。"""
    h = {0: 9, 1: 30}
    env = FakeEnv([60, 60])
    mc = FakeModelClient(lambda e, q: h[e])
    eval_one_episode_batch(env, mc)

    # env0: 60 步 / h=9 → 查询 7 次；env1: 60 步 / h=30 → 查询 2 次
    queries = {e: sum(e in envs for envs in mc.inference_envs()) for e in (0, 1)}
    assert queries == {0: 7, 1: 2}, queries

    # 每个 env 的动作里，query 编号恰好在自己的 h 边界处递增
    for env_idx, expected_h in h.items():
        steps = [s for e, s, _ in env.actions if e == env_idx]
        assert steps == list(range(60)), steps
        qnos = [a.split("s")[0] for e, _, a in env.actions if e == env_idx]
        boundaries = [i for i in range(1, 60) if qnos[i] != qnos[i - 1]]
        n_queries = -(-60 // expected_h)  # ceil(60 / h)
        assert boundaries == [expected_h * k for k in range(1, n_queries)], (
            env_idx, boundaries)


def test_heterogeneous_horizons_do_not_crosstalk():
    """各 env h 不同且互相不干扰：动作必须来自本 env 自己的 chunk。"""
    h = {0: 5, 1: 12, 2: 30}
    env = FakeEnv([40, 40, 40])
    mc = FakeModelClient(lambda e, q: h[e])
    eval_one_episode_batch(env, mc)

    for env_idx, step, action in env.actions:
        assert action.startswith(f"e{env_idx}q"), (env_idx, step, action)
    steps = {e: [s for i, s, _ in env.actions if i == e] for e in h}
    assert all(v == list(range(40)) for v in steps.values()), steps


def test_short_episode_mid_chunk_drops_buffer():
    """env 在 chunk 中途结束时，剩余缓冲被丢弃且不影响其它 env。"""
    env = FakeEnv([7, 30])
    mc = FakeModelClient(lambda e, q: 30)
    eval_one_episode_batch(env, mc)

    assert [s for e, s, _ in env.actions if e == 0] == list(range(7))
    assert [s for e, s, _ in env.actions if e == 1] == list(range(30))
    # env0 只查询过 1 次（chunk 30 > episode 7），env1 也是 1 次
    queries = {e: sum(e in envs for envs in mc.inference_envs()) for e in (0, 1)}
    assert queries == {0: 1, 1: 1}, queries


def test_obs_captured_every_step_for_all_running_envs():
    """每步都对全部 running env 取 obs —— render/capture/视频写盘依赖它。"""
    env = FakeEnv([10, 20])
    eval_one_episode_batch(env, FakeModelClient(lambda e, q: 30))

    # 共 20 步（max episode 长度）；每步一次，且覆盖当时全部 running env
    assert len(env.obs_calls) == 20, len(env.obs_calls)
    assert env.obs_calls[0] == (0, 1)
    assert env.obs_calls[-1] == (1,)
    assert all(tuple(sorted(c)) == (0, 1) for c in env.obs_calls[:10])


def test_empty_chunk_is_rejected():
    """空 chunk 必须显式报错，而不是静默空转。"""
    env = FakeEnv([5])
    try:
        eval_one_episode_batch(env, FakeModelClient(lambda e, q: 0))
    except AssertionError as exc:
        assert "empty action chunk" in str(exc), exc
    else:
        raise AssertionError("expected AssertionError for empty chunk")


if __name__ == "__main__":
    import traceback

    tests = [
        fn for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
