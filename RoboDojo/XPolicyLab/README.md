<div align="center">

<img src="https://xpolicylab.github.io/assets/logo.png" alt="XPolicyLab"/>

<p><strong>XPolicyLab: A unified standard and infrastructure for robot policy development and deployment.</strong></p>

</div>

XPolicyLab is the shared lane between policy code and evaluation environments. Keep each model's dependencies, checkpoints, and training recipes under `policy/<POLICY>/`; XPolicyLab handles the parts that are boring but easy to get wrong — serving, observation/action contracts, and eval wiring.

Start here for repo-level concepts and integration steps. For install commands, checkpoint layout, and training details, jump to that policy's README — it is the source of truth for its model.

## 🚀 What XPolicyLab Enables

- **Environment isolation**: run the policy model in its own conda/uv environment while the simulator, benchmark, or robot client runs separately.
- **Remote deployment**: connect the policy server and environment client through websocket, either on one machine or across machines.
- **A common adapter contract**: use the same high-level lifecycle for installation, data conversion, training, serving, and evaluation.
- **A large policy zoo**: reuse adapters for VLA/WAM policies, imitation-learning baselines, and reference templates.
- **Benchmark and infra integration**: mount XPolicyLab into benchmark or simulator workspaces without coupling policy code to one environment.

## 🌐 Supported Benchmarks And Infrastructure

**Benchmarks**

- **[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)**: supported for RoboDojo simulator-backed evaluation and RoboDojo-format data exports.
- **[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)**: supported as a benchmark and data source through policy-specific adapters and conversion scripts.

**Infrastructure**

- **RLinf**: supported infrastructure target for policy development and deployment workflows.
- **StarVLA**: supported infrastructure and policy stack; see [policy/starVLA](policy/starVLA/README.md).

## 🧭 Integrated Policies

Top-level adapters live in `policy/`. Treat each policy README as the source of truth for that model's paper/repo link, environment, data format, training entrypoint, and checkpoint layout.

<details>
<summary>Policy catalog</summary>

**Foundation / VLA / WAM policies**

- [A1](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/A1/README.md), [AHA_WAM](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/AHA_WAM/README.md), [Abot_M0](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Abot_M0/README.md), [Being_H05](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Being_H05/README.md), [Dexbotic_DM0](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Dexbotic_DM0/README.md), [Dexora_1B](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Dexora_1B/README.md)
- [DreamZero](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/DreamZero/README.md), [EventVLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/EventVLA/README.md), [FastWAM](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/FastWAM/README.md), [GO1](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/GO1/README.md), [GR00T_N17](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/GR00T_N17/README.md), [GalaxeaVLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/GalaxeaVLA/README.md)
- [GigaWorldPolicy](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/GigaWorldPolicy/README.md), [H_RDT](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/H_RDT/README.md), [Hy_Embodied_05_VLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Hy_Embodied_05_VLA/README.md), [InternVLA_A1](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/InternVLA_A1/README.md), [LDA_1B](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/LDA_1B/README.md)
- [LingBot_VA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/LingBot_VA/README.md), [LingBot_VLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/LingBot_VLA/README.md), [Mem_0](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Mem_0/README.md), [MolmoACT2](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/MolmoACT2/README.md), [Motus](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Motus/README.md)
- [OpenVLA_OFT](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/OpenVLA_OFT/README.md), [Pi_0](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Pi_0/README.md), [Pi_05](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Pi_05/README.md), [Pi_0_Fast](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Pi_0_Fast/README.md), [RDT_1B](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/RDT_1B/README.md), [RISE](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/RISE/README.md)
- [SmolVLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/SmolVLA/README.md), [Spatial_Forcing](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Spatial_Forcing/README.md), [Spirit_v15](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Spirit_v15/README.md), [TinyVLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/TinyVLA/README.md), [X_VLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/X_VLA/README.md), [X_WAM](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/X_WAM/README.md), [Xiaomi_Robotics_0](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Xiaomi_Robotics_0/README.md), [starVLA](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/starVLA/README.md)

**Baselines and examples**

- [ACT](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/ACT/README.md), [DP](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/DP/README.md), [demo_policy](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/demo_policy/README.md)

</details>

## 🧩 Framework Overview

XPolicyLab separates model-side dependencies from environment-side dependencies.

```text
Policy environment                         Evaluation / benchmark environment
------------------                         ----------------------------------
policy/<POLICY>/model.py     <---ws--->    env client / simulator / robot
policy server                              environment client
deploy.yml runtime config                  benchmark task and observation API
```

A typical adapter contains:

```text
policy/<POLICY>/
├── README.md                    # policy-specific guide
├── INSTALLATION.md              # optional detailed setup notes
├── install.sh                   # environment setup
├── process_data.sh              # optional data conversion
├── train.sh                     # optional training
├── eval.sh                      # same-machine evaluation
├── setup_eval_policy_server.sh  # policy-side server
├── setup_eval_env_client.sh     # environment-side client
├── deploy.yml                   # runtime config
├── deploy.py                    # evaluation loop
└── model.py                     # model adapter
```

`model.py` implements the model-facing API. `deploy.py` bridges environment observations to model-server calls. Use [policy/demo_policy](policy/demo_policy/README.md) as the minimal adapter reference.

`model.py` should define a `Model` class with this shape:

| Method | Contract |
| --- | --- |
| `__init__(model_cfg)` | Load model config, checkpoints, processors, and runtime overrides from `deploy.yml`. |
| `update_obs(obs)` | Update model state from one observation dictionary. |
| `update_obs_batch(obs_list)` | Update model state from a list of observation dictionaries. |
| `get_action()` | Return one action chunk as a list of action dictionaries. |
| `get_action_batch(env_idx_list=None)` | Return batched action chunks aligned with active environment indices. |
| `reset()` | Clear model-side state between evaluation episodes. |

The default policy-server protocol is websocket (`protocol: ws` in `deploy.yml`). Keep `legacy_tcp` only for old adapters that have not migrated yet.

## 🛠️ Model Integration Guide

The fastest way to add a model is to copy the reference adapter, keep the XPolicyLab boundary small, and debug the adapter before touching a real simulator.

1. **Learn the reference adapter**: read [policy/demo_policy](policy/demo_policy/README.md), especially `model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `setup_eval_policy_server.sh`, and `setup_eval_env_client.sh`.
2. **Understand the arguments**: keep `bench_name`, `task_name`, `ckpt_name`, `env_cfg_type`, `action_type`, and `seed` consistent across data, training, and eval.
3. **Create a skeleton**: run `bash scripts/create_policy.sh <POLICY_NAME>` and immediately fill in `policy/<POLICY_NAME>/README.md`.
4. **Implement `model.py` first**: load model resources in `__init__`, store observations in `update_obs`, translate observations to model-native inputs, return XPolicyLab action dictionaries from `get_action`, and reset state in `reset`.
5. **Keep deployment simple**: put runtime defaults in `deploy.yml`; keep `deploy.py` aligned with `demo_policy/deploy.py` unless the environment loop truly differs.
6. **Debug without a simulator**: run `EVAL_ENV_TYPE=debug` to check imports, server startup, observation serialization, action keys, action dimensions, and batch logic.
7. **Move to simulator or remote deployment**: after debug mode passes, use `EVAL_ENV_TYPE=sim` or split policy server and environment client across machines.

<details>
<summary>Agent Skill checklist for model integration</summary>

When using a coding agent, give it this checklist:

```text
Integrate <POLICY_NAME> into XPolicyLab.

Use policy/demo_policy as the reference.
1. Inspect the upstream model's inference API and dependencies.
2. Create or update policy/<POLICY_NAME>/README.md with install, checkpoint, train, and eval commands.
3. Implement install.sh and, if needed, process_data.sh and train.sh.
4. Implement model.py with Model.__init__, update_obs, get_action, reset, and batch methods.
5. Keep deploy.py aligned with policy/demo_policy/deploy.py.
6. Put runtime defaults in deploy.yml and use protocol: ws.
7. Run EVAL_ENV_TYPE=debug eval.sh and fix shape/action-key/server errors.
8. Summarize supported action_type, env_cfg_type, checkpoint layout, and remaining limitations.
```

A minimal Cursor Agent Skill can look like this:

```markdown
---
name: xpolicylab-model-integration
description: Guides agents through integrating a new robot policy into XPolicyLab. Use when adding or updating policy/<POLICY>/ adapters, model.py, deploy.py, deploy.yml, install scripts, training scripts, or debug-mode evaluation.
---

# XPolicyLab Model Integration

Follow the XPolicyLab README and policy/demo_policy reference adapter.

1. Read policy/demo_policy/model.py, deploy.py, deploy.yml, eval.sh, and README.md.
2. Inspect the target model's inference API, dependencies, checkpoint layout, and expected observations/actions.
3. Create or update policy/<POLICY>/ with scripts/create_policy.sh if needed.
4. Implement model.py first; keep upstream model code unchanged when possible.
5. Keep deploy.py aligned with demo_policy unless the environment loop truly differs.
6. Put runtime defaults in deploy.yml and prefer protocol: ws.
7. Run EVAL_ENV_TYPE=debug before simulator-backed evaluation.
8. Document exact install, train, eval, action_type, env_cfg_type, and checkpoint assumptions in policy/<POLICY>/README.md.
```

</details>

## ⚡ Quick Start

Clone XPolicyLab as a normal Python project when you are developing adapters, running offline checks, training from prepared data, or using your own environment client:

```bash
mkdir demo_env
cd demo_env
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
pip install -e .
```

You do not need a simulator installation to start model-side development. The downloader in this standalone XPolicyLab repo fetches prepared RoboDojo data for XPolicyLab-only training and offline debugging, without requiring a RoboDojo simulator checkout. It provides multiple RoboDojo data versions, plus HDF5 `RoboDojo_real` data for real-world experiments.

If you are using the `XPolicyLab/` subpackage inside the RoboDojo repository, follow RoboDojo's own data download scripts instead of this standalone helper.

Download a small Hugging Face demo bundle and keep the data next to `XPolicyLab/`:

```bash
# From demo_env/XPolicyLab
bash scripts/RoboDojo/download_robodojo_data.sh demo
```

This creates:

```text
demo_env/
├── data/        # demo data, including a small 10-episode HuggingFace bundle
└── XPolicyLab/
```

You can also pull HDF5 or LeRobot exports for RoboDojo and other benchmark-backed experiments:

```bash
# RoboDojo HDF5 data, saved to ../data/RoboDojo
bash scripts/RoboDojo/download_robodojo_data.sh hdf5

# RoboDojo LeRobot v3.0 video data, saved to ../data/RoboDojo_lerobot_v30_video
bash scripts/RoboDojo/download_robodojo_data.sh lerobot_v3.0

# RoboDojo LeRobot v2.1 video data, saved to ../data/RoboDojo_lerobot_v21_video
bash scripts/RoboDojo/download_robodojo_data.sh lerobot_v2.1

# RoboDojo real-world HDF5 data, saved to ../data/RoboDojo_real
bash scripts/RoboDojo/download_robodojo_data.sh real
```

With this setup, you can test data conversion, model loading, training scripts, and debug-mode evaluation before connecting to a simulator-backed benchmark.

```bash
export EVAL_ENV_TYPE=debug
cd policy/demo_policy
pip install -e .
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 base base
```

The template for any adapter is the same — swap `demo_policy` and the argument values:

```bash
export EVAL_ENV_TYPE=debug
cd policy/<POLICY>
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

For RoboDojo simulation, mount `XPolicyLab/` beside the simulator-side `env_cfg/`, `scripts/`, `src/eval_client/`, and `task/` directories.

## 🔄 Common Workflow

Most adapters expose the same top-level shape. Some policies add extra arguments, consume upstream-native datasets, or skip training support. Follow the policy README when it differs from this template.

```bash
cd policy/<POLICY>

# Install the policy runtime.
bash install.sh

# Optional: convert or prepare policy-specific data.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]

# Optional: train.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Evaluate on one machine.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

### What the arguments mean

When you run `eval.sh`, you are mostly answering: **which benchmark family**, **which task to run now**, **which checkpoint to load**, **which robot setup**, **joint or end-effector actions**, and **which seed**. The same names travel through `process_data.sh`, `train.sh`, and `eval.sh`, so you do not have to rename things at every step.

| Argument | In plain English | Examples |
| --- | --- | --- |
| `bench_name` | Which benchmark or dataset family this run belongs to | `RoboDojo`, `RoboTwin` |
| `task_name` | The task the environment client should run right now | `stack_bowls`, `push_T` — can differ from the tasks seen during training |
| `ckpt_name` | Which weights to load: a short run nickname, the full run folder name, or a path | `cotrain`, `RoboDojo-cotrain-arx_x5-joint-0`, `checkpoints/my_run/` |
| `env_cfg_type` | Robot / camera / scene configuration key | `arx_x5` |
| `action_type` | Action space the policy outputs | usually `joint` or `ee` |
| `seed` | Training or evaluation seed / layout id | `0`, `1`, `2` |
| `policy_gpu_id` / `env_gpu_id` | Which GPU runs the model vs. the simulator/client | `0`, `1` |
| `policy_env_or_uv_path` | Conda env name or uv env path for the policy server | your policy-side env |
| `eval_env_conda_env` | Conda env for the simulator / robot client | your eval-side env |

**How `ckpt_name` resolves.** Most of the time you pass the short nickname you used during training, such as `cotrain`. XPolicyLab combines it with the other args and looks under `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`. Already know the full folder name? Pass that instead. Weights live somewhere else? Pass a path — relative paths resolve from the policy directory, absolute paths work too. Some adapters also honor explicit keys in `deploy.yml` (`checkpoint_path`, `model_path`, ...). When in doubt, check the policy README.

**A concrete eval example:**

```bash
cd policy/AHA_WAM
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 aha_wam robodojo
# loads checkpoints/RoboDojo-cotrain-arx_x5-joint-0/ and evaluates on stack_bowls
```

## 🔌 Deployment Flow

During evaluation, the policy server and the environment client talk over websocket. That split is what lets you keep Isaac Sim / robot drivers on one machine and a heavy VLA on another.

For same-machine evaluation, `eval.sh` is enough — it starts the server, runs the client, and cleans up when you are done.

For split-machine deployment, start the policy server on the GPU machine and bind to `0.0.0.0` so other machines can reach it. The client connects to the policy machine's real IP, not `0.0.0.0`.

```bash
cd policy/<POLICY>
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_env_or_uv_path> <policy_server_port> 0.0.0.0
```

Then start the environment client on the simulator or robot machine:

```bash
cd policy/<POLICY>
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

`EVAL_ENV_TYPE` selects the environment-side backend:

- unset or `sim`: real simulator-backed evaluation, when the integration is installed.
- `debug`: offline wiring check — no Isaac, no robot, just shapes and IO.
- `real`: real-robot client path, where the hardware integration exists.

## 📐 Standard Data Formats

XPolicyLab standardizes the observation and trajectory dictionaries passed between adapters, converters, and environment clients. Individual policies may convert this standard format into their upstream-native format.

All pose values use `[x, y, z, qw, qx, qy, qz]`. Images are RGB unless a policy README states otherwise.

<details>
<summary>Observation Data Format v1.0</summary>

```text
Observation Data Format v1.0
├── data_format_version                        string, optional
├── instruction / instructions                 string or list[str]
├── env_idx                                    int, optional for batched eval
├── additional_info/
│   └── frequency                              int, optional
├── vision/
│   ├── cam_head/
│   │   ├── color                              (H, W, 3), RGB
│   │   ├── depth                              (H, W) or (H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3), optional
│   │   ├── extrinsics_matrix                  (4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
└── state/
    ├── left_arm_joint_state                   (DOF,), optional
    ├── left_ee_joint_state                    (EEF_DOF,), optional
    ├── left_ee_pose                           (7,), optional
    ├── left_tcp_pose                          (7,), optional
    ├── left_delta_ee_pose                     (7,), optional
    ├── right_arm_joint_state                  (DOF,), optional
    ├── right_ee_joint_state                   (EEF_DOF,), optional
    ├── right_ee_pose                          (7,), optional
    ├── right_tcp_pose                         (7,), optional
    ├── right_delta_ee_pose                    (7,), optional
    ├── arm_joint_state                        (DOF,), optional for single-arm robots
    ├── ee_joint_state                         (EEF_DOF,), optional for single-arm robots
    ├── ee_pose                                (7,), optional for single-arm robots
    ├── tcp_pose                               (7,), optional for single-arm robots
    ├── delta_ee_pose                          (7,), optional for single-arm robots
    └── mobile/                                optional
        ├── base_pose                          (7,)
        └── base_twist                         (6,), [vx, vy, vz, wx, wy, wz]
```

</details>

<details>
<summary>Trajectory Data Format v1.0</summary>

```text
Trajectory Data Format v1.0
├── data_format_version                        string, e.g. "v1.0"
├── instructions                               JSON-serialized list[str]
├── subtasks                                   JSON-serialized annotations, optional
├── additional_info/
│   └── frequency                              int
├── vision/
│   ├── cam_head/
│   │   ├── colors                             (T, H, W, 3), uint8 RGB or encoded stream
│   │   ├── depths                             (T, H, W) or (T, H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3) or (T, 3, 3), optional
│   │   ├── extrinsics_matrix                  (4, 4) or (T, 4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
└── state/
    ├── left_arm_joint_states                  (T, DOF), optional
    ├── left_ee_joint_states                   (T, EEF_DOF), optional
    ├── left_ee_poses                          (T, 7), optional
    ├── left_tcp_poses                         (T, 7), optional
    ├── left_delta_ee_poses                    (T, 7), optional
    ├── right_arm_joint_states                 (T, DOF), optional
    ├── right_ee_joint_states                  (T, EEF_DOF), optional
    ├── right_ee_poses                         (T, 7), optional
    ├── right_tcp_poses                        (T, 7), optional
    ├── right_delta_ee_poses                   (T, 7), optional
    ├── arm_joint_states                       (T, DOF), optional for single-arm robots
    ├── ee_joint_states                        (T, EEF_DOF), optional for single-arm robots
    ├── ee_poses                               (T, 7), optional for single-arm robots
    ├── tcp_poses                              (T, 7), optional for single-arm robots
    ├── delta_ee_poses                         (T, 7), optional for single-arm robots
    └── mobile/                                optional
        ├── base_poses                         (T, 7)
        └── base_twists                        (T, 6), [vx, vy, vz, wx, wy, wz]
```

</details>

Useful converter helpers:

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
```

`decode_image_bit` handles encoded RGB image streams. `get_robot_action_dim_info(env_cfg_type)` returns robot-specific `arm_dim` and `ee_dim` lists, so adapters do not need to hard-code action dimensions.

Robot action dimensions are registered in `utils/robot/_robot_info.json`. Each top-level key is an `env_cfg_type` such as `arx_x5`, and its `arm_dim` / `ee_dim` lists describe the per-arm joint dimensions and end-effector or gripper dimensions. Update this file when adding a new robot configuration so data conversion, training, and deployment code can infer action shapes consistently.

## 💾 Data And Checkpoints

Training and data prep usually name things predictably so eval can find them without guesswork:

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
```

So if you trained with `bench_name=RoboDojo`, `ckpt_name=cotrain`, `env_cfg_type=arx_x5`, `action_type=joint`, `seed=0`, the run lands in `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`. At eval time you can pass just `cotrain` and let XPolicyLab stitch the rest together — or pass the full folder name, or a direct path if your weights live elsewhere.

Policies may also use upstream-native layouts or explicit paths in `deploy.yml`. Check the policy README before assuming a naming convention. For a small local dataset to play with, see [Quick Start](#-quick-start).

## ✅ Checks

Run static checks from the XPolicyLab repo root. Run `eval.sh` from `policy/<POLICY>/`, same as [Common Workflow](#-common-workflow).

Static checks:

```bash
git diff --check
bash -n policy/<POLICY>/*.sh
python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py
```

Adapter wiring check (no simulator required):

```bash
pip install -e .
export EVAL_ENV_TYPE=debug
cd policy/<POLICY>
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 \
  <policy_env_or_uv_path> <eval_env_conda_env>
```

For a quick smoke test, try `policy/demo_policy` with placeholder env names such as `base`. Argument details live in [Common Workflow](#-common-workflow).

## 📬 Contact

Tianxing Chen: [chentianxing2002@gmail.com](mailto:chentianxing2002@gmail.com)
