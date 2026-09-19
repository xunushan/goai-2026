# patch_policy（flow_policy 部署适配器）

`patch_policy` 是 **flow_policy**（DINOv2 + patch_policy transformer 骨干 +
X-VLA flow matching，训练仓库 `github.com/xunushan/flow-policy.git` 的 `flow_policy/`
子目录）在 XPolicyLab 下的策略服务适配器。

## 架构

```
输入: 3 路相机图像 (cam_head/cam_left_wrist/cam_right_wrist) + 20 维 ee6d proprio
  ├─ 图像: HWC → Resize(224,224) BICUBIC → [0,1] CHW，在模型内 DINOv2 编码
  │        （训练 precompute 同款预处理，无增强、无 Normalize——DINOv2 内部做）
  ├─ proprio: 由 ObsManager state 构建（left/right ee pose + gripper，不反转，1=开）
  └─ FlowPolicy.generate_actions(steps)  →  30 步 flow matching 去噪
输出: [num_actions, 20] arx_ee6d action chunk
      [l_xyz(3), l_rot6d(6), l_g(1), r_xyz(3), r_rot6d(6), r_g(1)]
      → 转回 16 维 ee dict 协议（xyz + quat_wxyz + gripper）
```

模型无语言/domain 概念，忽略 instruction。

## 目录结构

```
patch_policy/
├── model.py              # 策略服务核心（观测→推理→动作协议）
├── deploy.yml            # 服务配置（ckpt 路径、模型超参、steps/actions_per_chunk 等）
├── flow_policy/models/   # vendored 模型代码（models 包，自包含；vision.py 已做离线加固）
├── deploy.py / eval.sh / setup_eval_*.sh   # RoboDojo 集成入口（与 X_VLA 一致）
├── install.sh            # 校验依赖（复用 XVLA conda 环境，不安装新包）
└── serve_remote.sh       # 便捷启动脚本
```

`flow_policy/models/` 是从训练仓库复制的模型包（`model.py` / `vision.py` /
`transformer.py` / `action_hub.py`）。`vision.py` 的 DINOv2 加载改为
`pretrained=False`（架构与训练一致，权重由 ckpt 的 `model_state_dict` 灌回），
避免服务机从 `dl.fbaipublicfiles.com` 下载预训练权重。

## 部署（issac-server）

issac-server 上模型在 `/data/goai_800/checkpoints/`，conda 环境 `XVLA`
（torch 2.1.2+cu121，含 einops/scipy/PIL/cv2 等推理依赖）。`/data/RoboDojo`
为 RoboDojo 主仓库（含 XPolicyLab/policy）。

1. 同步本目录到服务器 `/data/RoboDojo/XPolicyLab/policy/patch_policy`（上传复制）。
2. 校验依赖：`bash install.sh`（复用 XVLA 环境，不安装）。
3. 确认 `deploy.yml` 参数（ckpt 路径 / steps / actions_per_chunk / camera_names / port）。
4. 启动服务（screen 长驻，日志到 `/data/outputs`，参考策略服务启动指南）：
   ```bash
   cd /data/RoboDojo
   bash scripts/robodojo.sh server \
     --policy-dir XPolicyLab/policy/patch_policy \
     --task <task_name> \
     --ckpt /data/goai_800/checkpoints/ckpt-5000.pt \
     --policy-env XVLA \
     --env-cfg arx_x5 \
     --action-type ee \
     --seed 0 \
     --policy-gpu 0 \
     --policy-port <port> \
     --bind-host <host>
   ```

## 关键配置（deploy.yml）

- `ckpt_path`：flow_policy checkpoint 文件（.pt），可被 `--ckpt` 覆盖。
- `steps`：flow matching 去噪迭代次数（默认 10）。
- `actions_per_chunk`：一次重规划后执行的动作数（默认 30 = 整段直出）。
- `camera_names`：视角顺序，默认 `[cam_head, cam_left_wrist, cam_right_wrist]`。
- `gripper_mode`：`continuous`（直出 [0,1]）或 `threshold`（二值化）。
- `model`：模型超参，必须与训练 `flow_policy/configs/train.yaml` 一致。
