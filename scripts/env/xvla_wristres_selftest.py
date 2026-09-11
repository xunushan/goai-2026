"""R0/R1 腕部残差「服务端适配」上线自检。

在 issac-server 的 XVLA conda env 下运行（cwd 任意）：

    cd /data/RoboDojo && \
    python /workspace/xvla_wristres_selftest.py

覆盖 7 项：
  T1 路由：policy_model_class 未配置 → 返回的就是 XVLA 本体（对象恒等），旧路径零改动
  T2 路由：配置为 wrist_action_residual 路径 → 返回 WristActionResidualXVLA
  T3 防呆：R0 ckpt + 未配置 → 明确报错（不静默丢残差分支）
  T4 防呆：基础 ckpt + 已配置 → 明确报错（不静默用错类）
  T5 加载：R0 / R1 ckpt 用残差类加载，wrist_residual.* 权重非零且全有限
  T6 推理：R0 / R1 generate_actions 返回 (1, num_actions, 20) 且全有限
  T7 回归：基础 ckpt 经新 Model 路径加载 vs 直接 XVLA 加载，
          generate_actions 在固定 x1 下逐元素 bit 级一致（证明路由改动零漂移）
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

POLICY_DIR = Path("/data/RoboDojo/XPolicyLab/policy/X_VLA")
sys.path.insert(0, str(POLICY_DIR))

import model as xvla_policy  # noqa: E402
from xvla.models.modeling_xvla import XVLA  # noqa: E402

BASE_CKPT = "/workspace/x0_ee6d_sim_fwloss/pretrained/ckpt-20000"
R0_CKPT = "/workspace/xvla_r0/pretrained/ckpt-4000"
R1_CKPT = "/workspace/xvla_r1/pretrained/ckpt-4000"

WRIST_PATH = xvla_policy.WRIST_RESIDUAL_DOTTED_PATH
DEVICE = torch.device("cuda")

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)


def base_cfg(**overrides) -> dict:
    cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
    cfg.update(device="cuda", env_cfg_type=None, env_cfg=None, seed=0)
    cfg.update(overrides)
    return cfg


class _ResolverStub:
    """只带 model_cfg 的壳，用于单独调用路由解析（避免完整加载模型）。"""

    def __init__(self, cfg: dict) -> None:
        self.model_cfg = cfg

    _resolve_policy_model_class = xvla_policy.Model._resolve_policy_model_class


def expect_raises(name: str, fn, *fragments: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - 自检脚本要在报告里原样呈现异常
        text = str(exc)
        missing = [f for f in fragments if f not in text]
        if missing:
            record(name, False, f"报错信息未包含 {missing}；实际: {text[:300]}")
        else:
            record(name, True, f"{type(exc).__name__}")
        return
    record(name, False, "预期报错但正常返回")


def synth_obs(camera_names: list[str], seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "images": [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in camera_names],
        "proprio": np.zeros(20, dtype=np.float32),
        "prompt": "selftest",
    }


def main() -> int:
    # ---------- T1 / T2 / T3 / T4：路由与防呆（轻量，不加载权重）----------
    resolver = _ResolverStub(base_cfg())

    model_cls = resolver._resolve_policy_model_class(BASE_CKPT)
    record(
        "T1 未配置 policy_model_class → 返回 XVLA 本体",
        model_cls is XVLA,
        f"got {model_cls.__name__}",
    )

    resolver_on = _ResolverStub(base_cfg(policy_model_class=WRIST_PATH))
    model_cls = resolver_on._resolve_policy_model_class(R0_CKPT)
    record(
        "T2 配置 policy_model_class → 返回 WristActionResidualXVLA",
        model_cls.__name__ == "WristActionResidualXVLA",
        f"got {model_cls.__name__}",
    )

    expect_raises(
        "T3 防呆：R0 ckpt + 未配置 → 报错",
        lambda: resolver._resolve_policy_model_class(R0_CKPT),
        "declares a wrist residual model",
        "policy_model_class",
    )
    expect_raises(
        "T4 防呆：基础 ckpt + 已配置 → 报错",
        lambda: resolver_on._resolve_policy_model_class(BASE_CKPT),
        "does not declare a wrist residual model",
    )

    # ---------- T5 / T6：R0 / R1 真实加载与推理 ----------
    cams3 = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    for tag, ckpt in (("R0", R0_CKPT), ("R1", R1_CKPT)):
        cfg = base_cfg(
            model_path=ckpt,
            policy_model_class=WRIST_PATH,
            camera_names=cams3,
        )
        net = xvla_policy.Model(cfg)
        loaded = net.model
        record(
            f"T5a {tag} 加载出的类正确",
            type(loaded).__name__ == "WristActionResidualXVLA",
            type(loaded).__name__,
        )
        wrist = {
            k: v for k, v in loaded.state_dict().items() if k.startswith("wrist_residual.")
        }
        finite = all(bool(torch.isfinite(v).all()) for v in wrist.values())
        nonzero = any(float(v.abs().max()) > 0 for v in wrist.values())
        record(
            f"T5b {tag} wrist_residual.* 权重有限且非零",
            bool(wrist) and finite and nonzero,
            f"n={len(wrist)} finite={finite} nonzero={nonzero} "
            f"max_abs={max(float(v.abs().max()) for v in wrist.values()):.4g}",
        )
        mode = getattr(loaded.config, "wrist_residual_mode", None)
        has_gate = getattr(loaded.wrist_residual, "arm_gate", None) is not None
        record(
            f"T5c {tag} mode/arm_gate 与 config 一致",
            mode == tag.lower() and has_gate == (tag == "R1"),
            f"mode={mode} arm_gate={has_gate}",
        )

        obs = synth_obs(cams3)
        with torch.no_grad():
            actions = net.infer(obs, steps=10)
        record(
            f"T6 {tag} generate_actions 输出形状/有限性",
            actions.shape == (30, 20) and bool(np.isfinite(actions).all()),
            f"shape={actions.shape} finite={bool(np.isfinite(actions).all())}",
        )
        del net, loaded
        torch.cuda.empty_cache()

    # ---------- T7：基础 ckpt 路径零漂移（逐元素一致）----------
    cfg_base = base_cfg(model_path=BASE_CKPT, camera_names=["cam_head"])
    net_new = xvla_policy.Model(cfg_base)
    inputs = net_new.processor(
        images=[Image.fromarray(img) for img in synth_obs(["cam_head"])["images"]],
        language_instruction="selftest",
    )
    inputs = {
        k: (v.to(device=DEVICE, dtype=torch.float32) if v.is_floating_point() else v.to(DEVICE))
        for k, v in inputs.items()
    }
    inputs["proprio"] = torch.zeros(1, 20, dtype=torch.float32, device=DEVICE)
    inputs["domain_id"] = torch.zeros(1, dtype=torch.long, device=DEVICE)

    num_actions = int(net_new.model.num_actions)
    dim_action = int(net_new.model.action_space.dim_action)
    x1 = torch.randn(1, num_actions, dim_action, device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        got_new = net_new.model.generate_actions(**inputs, steps=10, x1=x1)
    del net_new
    torch.cuda.empty_cache()

    net_ref = (
        XVLA.from_pretrained(BASE_CKPT, trust_remote_code=True, torch_dtype=torch.float32)
        .to(DEVICE)
        .to(torch.float32)
    )
    with torch.no_grad():
        got_ref = net_ref.generate_actions(**inputs, steps=10, x1=x1)
    del net_ref
    torch.cuda.empty_cache()

    identical = torch.equal(got_new, got_ref)
    record(
        "T7 基础 ckpt：新 Model 路径 vs 直接 XVLA 逐元素一致",
        identical,
        f"max_abs_diff={float((got_new - got_ref).abs().max()):.3e}",
    )

    print("\n===== 自检汇总 =====", flush=True)
    failed = [name for name, ok, _ in RESULTS if not ok]
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    print(f"总计 {len(RESULTS)} 项，失败 {len(failed)} 项", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
