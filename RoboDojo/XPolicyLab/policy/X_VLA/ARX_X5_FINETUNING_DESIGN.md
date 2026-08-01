# X-VLA 在 RoboDojo ARX-X5 LeRobot v3 EE 数据上的微调方案

## 1. 目标与已确认的数据定义

本文档定义如何使用 `policy/X_VLA/train.sh` 的原生训练框架，在 RoboDojo
ARX-X5 双臂数据上微调 X-VLA。

当前数据集已经确认具有以下属性：

| 项目 | 定义 |
|---|---|
| 数据格式 | LeRobot v3.0，Parquet + MP4 |
| 采集频率 | 25 Hz |
| 相机 | `cam_high`、`cam_left_wrist`、`cam_right_wrist`，逐帧同步 |
| 状态和动作 | 双臂绝对 EE，原始维度 16 |
| 单臂布局 | `xyz(3) + quaternion_wxyz(4) + continuous_gripper(1)` |
| X-VLA 内部布局 | 双臂 EE6D，维度 20 |
| 轨迹长度 | 30 个未来 action anchor |
| 轨迹物理窗口 | 1 秒 |
| 初始 domain | `domain_id=6`，复用 RoboTwin2/ARX-X5 domain |
| 夹爪监督 | 连续值，MSE |

第一阶段的原则是尽量复用 `domain_id=6` 已经学习到的机器人平台和时间语义，
只对数据格式、相机数量和连续夹爪做必要适配。新增 domain 留作后续独立实验。

## 2. 数据接入：新增 Handler，不新增旁路 Dataset

### 2.1 设计决定

在 X-VLA 的 `datasets/domain_handler/` 下新增：

```text
RoboDojoLeRobotV3EEHandler(DomainHandler)
```

并在 `datasets/domain_handler/registry.py` 中以独立数据集名注册，例如：

```python
"RoboDojo_LerobotV3_ARX_EE": RoboDojoLeRobotV3EEHandler
```

不使用之前实验性旁路实现中的独立 PyTorch `Dataset` 训练入口，但复用其中已经
验证过的 LeRobot v3 metadata、episode 和视频读取思路。新 Handler 必须继续接入
X-VLA 原有链路：

```text
meta JSON
  -> InfiniteDataReader
  -> handler registry
  -> RoboDojoLeRobotV3EEHandler.iter_episode()
  -> action_slice()
  -> X-VLA train.py
```

这样可以保留 X-VLA 原生的多数据集采样、语言增强、domain 路由和训练流程。

### 2.2 复用现有公共函数

不重新实现四元数转换和图像预处理：

- wxyz quaternion 转 rotation-6D：复用
  `xvla/datasets/utils.py::quat_wxyz_to_rotate6d()`；
- rotation-6D 转 quaternion：部署时复用
  `xvla/datasets/utils.py::rotate6d_to_quat(..., scalar_first=True)`；
- resize、ColorJitter、ImageNet normalization：复用
  `InfiniteDataReader.image_aug`；
- 当前状态和未来动作切分：复用 `xvla/datasets/utils.py::action_slice()`。

Handler 只增加 ARX-X5 字段编排，不重复实现旋转算法：

```text
LeRobot 16D
  left:  xyz + wxyz + gripper
  right: xyz + wxyz + gripper
             |
             v
X-VLA 20D
  left:  xyz + rotation6d + gripper
  right: xyz + rotation6d + gripper
```

夹爪是否需要 `1 - value` 必须以数据集 metadata/采集定义为准，并在 Handler
和部署 adapter 中成对实现。禁止仅根据变量名猜测 open/close 方向。

### 2.3 三相机处理

Handler 按固定顺序返回三路图像：

```python
[
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
```

输出形状：

```text
image_input: [3, C, 224, 224]
image_mask:  [True, True, True]
```

第 0 路必须是 `cam_high`：它进入 Florence-2 的主视觉—语言融合路径；两路腕部
图像经视觉编码器后作为 action transformer 的辅助视觉 token。由于三路视频已逐帧
同步，当前图像统一使用同一 episode 内的同一个 frame index。

## 3. 轨迹时间定义与 30 个 anchor

### 3.1 `freq` 必须使用真实的 25 Hz

不能把 `freq` 写成 30 来匹配 `num_actions=30`。`freq` 的作用是把原始帧编号转换
为物理时间：

```python
source_time = np.arange(num_frames) / freq
```

如果把真实 25 Hz 数据错误写成 30 Hz，第 25 帧会被解释为 `0.8333s`，但它实际
发生在 `1.0s`。这会压缩整段轨迹的时间轴，并从错误的未来时刻构造监督信号。

确定配置为：

```python
freq = 25.0
qdur = 1.0
num_actions = 30
```

### 3.2 anchor 的生成方式

对于当前帧真实时间 `cur`，构造包含当前点在内的 31 个查询时间：

```python
q = np.linspace(cur, cur + 1.0, 31, dtype=np.float32)
```

对应关系为：

```text
q[0]      -> 当前 proprio
q[1:31]   -> 未来 30 个 action anchor
```

原始 25 Hz 轨迹在一秒内包含 25 个未来采样间隔。X-VLA 将这条连续轨迹表示为
30 个固定长度 token，因此这里是 25 到 30 的轻微时间插值，而不是论文所举的
4 秒高频轨迹到 30 点的降采样示例。

这样选择的原因是当前阶段使用 `domain_id=6`。RoboTwin2 handler 将该 domain 的
30 个输出定义为约 1 秒未来轨迹。保持 `qdur=1.0` 可以保留已有 domain 的时间
语义；将其直接改为 4 秒会同时引入新的时间跨度 domain shift。

论文中 temporal downsampling 的核心机制仍然成立：不同频率、不同长度的轨迹被
映射为固定 30 个 anchor。源码本身并未对所有数据统一使用 4 秒，多个 domain 使用
了 1、2、4、5 或 10 秒窗口。4 秒版本可以在新增独立 ARX domain 后作为消融实验，
不纳入当前第一阶段方案。

### 3.3 episode 末尾不能压缩时间窗口

当前通用 `BaseHDF5Handler` 会执行：

```python
min(cur + qdur, episode_end)
```

这会把 episode 尾部不足 1 秒的剩余轨迹仍然压成 30 个点，使不同样本的 anchor
时间间隔不一致。新 Handler 不采用这种处理。

候选当前帧必须满足：

```python
cur + qdur <= episode_end
```

不足完整 1 秒窗口的尾部样本直接排除，不截短、不端点复制，也不把更短轨迹压缩
为 30 个点。

### 3.4 训练与部署的时间语义必须一致

模型不显式接收 `freq` 或 `qdur`，只输出 `[30, 20]`。因此 `domain_id=6` 和训练
数据共同隐式定义：30 个点覆盖未来 1 秒。

ARX-X5 控制端为 25 Hz，不能把 30 个预测点按 25 Hz 全部逐点执行，否则预测的
1 秒轨迹会被执行成 1.2 秒。部署端采用以下二者之一：

1. 将包含当前 proprio 的 31 点预测轨迹按时间重采样为未来 25 个控制点；
2. 只执行前若干个 25 Hz 控制点，然后重新获取观测并规划。

第一版建议实现明确的 `30 anchors / 1s -> 25 control steps / 1s` 重采样，再通过
`actions_per_chunk` 控制实际执行多少个控制点。位置和连续夹爪使用线性插值；旋转
在部署边界转换为 quaternion 后使用 SLERP。

## 4. Domain 适配方案

### 4.1 当前方案

训练和推理统一使用：

```text
domain_id = 6
```

理由：RoboTwin2 同样使用 ARX-X5 双臂平台，绝对 EE 动作布局和控制场景与当前
RoboDojo 数据更接近。domain ID 用于区分机器人/domain，而不是具有数值距离的类别。

当前阶段直接复用 ID 6 的以下参数：

- soft prompt；
- domain-specific action encoder；
- domain-specific action decoder；
- 配置启用时的其他 domain-aware projection。

训练脚本的 prompt warm-up 阶段按照现有 X-VLA finetuning 逻辑执行：训练当前
domain 的 soft prompt 和 action encoder/decoder，同时冻结 VLM 与 transformer
core；之后进入联合训练阶段。

### 4.2 后续新增 domain

新增 ARX domain 不属于当前实现范围。后续实验时应：

1. 审计预训练数据和 checkpoint，确认选定 ID 确实未被使用；
2. 为新硬件 domain 使用新的 learnable prompt/domain-specific action 层；
3. 按 X-VLA 的两阶段适配流程训练；
4. 不因为机器人相近而默认复制 ID 6 参数，是否复制仅作为单独对照实验。

## 5. 连续夹爪与自定义 action space

新增 action space：

```text
action_mode = "arx_ee6d"
dim_action = 20
gripper_idx = (9, 19)
```

参考 `AGIBOTEE6DActionSpace`，但使用独立名称和配置，避免将 ARX-X5 行为绑定到
AGIBOT 的后续改动。

确定行为：

- xyz：MSE；
- rotation-6D：MSE；
- continuous gripper：MSE；
- `preprocess()` 保留 proprio 和 noisy action 中的夹爪通道；
- `postprocess()` 不使用 sigmoid；
- 若数据约定为 `[0,1]`，推理输出最终 clip 到 `[0,1]`；
- 首版沿用 AGIBOT 的 loss scale 作为起点，训练时分别记录三个 loss 的量级，再决定
  是否调整夹爪权重。

模型内部继续使用 20 维 rotation-6D，不把 action head 改成原始 16 维 quaternion，
从而保留预训练 action transformer/head 的结构兼容性，并避免 quaternion 的正负二义性。

注意：`train.sh` 的 `action_type` 当前只参与实验目录命名。真正选择 action space 的是
模型 `config.json` 中的 `action_mode`，训练入口必须显式验证其为 `arx_ee6d`，避免
静默使用默认 `ee6d` 的 BCE 夹爪损失。

## 6. Episode 列表的配置与处理位置

### 6.1 职责边界

Episode 筛选属于数据配置，不属于模型或训练循环。处理位置确定为：

```text
meta 配置声明 episode 选择
  -> InfiniteDataReader 加载 meta 时解析和验证
  -> datalist 保存确定的 episode ID
  -> Handler.iter_episode(traj_idx) 映射到具体 episode ID
```

训练脚本不直接展开 episode，不在 batch 生成后过滤，也不按 Parquet 文件筛选。
LeRobot v3 的一个 Parquet/video chunk 可以包含多个 episode，筛选单位必须是
`episode_index`，否则可能跨 episode 构造未来 action。

### 6.2 推荐的外部 split 文件

推荐使用独立 JSON 保存固定划分：

```json
{
  "train": [0, 1, 2, 5, 8],
  "val": [3, 4],
  "test": [6, 7]
}
```

meta 中引用：

```json
{
  "dataset_name": "RoboDojo_LerobotV3_ARX_EE",
  "dataset_root": "/absolute/path/to/lerobot_v30_ee",
  "repo_id": "robodojo_arx_ee",
  "episode_split_path": "/absolute/path/to/arx_ee_split.json",
  "episode_split": "train",
  "domain_id": 6,
  "fps": 25,
  "query_duration": 1.0,
  "observation_key": [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist"
  ]
}
```

也允许小规模调试时内联：

```json
"episodes": [0, 1, 2, 5, 8]
```

`episodes` 与 `episode_split_path` 必须互斥。默认不允许两者都缺失；确实要训练全部
episode 时要求显式配置：

```json
"allow_all_episodes": true
```

### 6.3 加载时验证

meta 加载阶段必须完成以下检查：

- split 文件存在且指定 split key 存在；
- episode ID 为非负整数；
- 列表非空且没有重复；
- ID 存在于 LeRobot metadata；
- train/val/test 划分之间无交集；
- 选定 episode 至少有一个满足完整 1 秒未来窗口的样本；
- 记录最终 episode 数量和 frame/sample 数量。

每个 DataLoader worker 各自初始化只读 LeRobot reader，避免在 worker 间共享不可安全
序列化的视频解码状态。Handler 实例缓存 reader 和 episode 到帧范围的映射，不能在
每个 sample 上重复初始化整个 LeRobot dataset。

### 6.4 可复现性

开始训练时，将以下文件复制到输出目录：

- 完整解析后的训练 meta；
- 原始 episode split JSON；
- 最终展开后的 episode ID 列表；
- 数据集 `meta/info.json` 的副本或摘要；
- action/domain/frequency/query-duration 配置。

日志至少输出：

```text
dataset_root
dataset_version
fps
qdur
num_actions
domain_id
action_mode
camera_keys
selected_episode_count
valid_sample_count
```

## 7. 需要修改的文件

计划改动范围：

1. `xvla/datasets/domain_handler/lerobot_v3_robodojo.py`
   - 新增 ARX-X5 LeRobot v3 Handler；
   - 三相机读取；
   - 16D wxyz 到 20D rotation-6D；
   - 25 Hz、1 秒、30 anchor；
   - episode 边界检查。
2. `xvla/datasets/domain_handler/registry.py`
   - 注册新 dataset name。
3. `xvla/datasets/dataset.py`
   - 解析显式 domain ID；
   - 解析 episode/split 配置；
   - 为 LeRobot Handler 建立 episode datalist。
4. `xvla/models/action_hub.py`
   - 注册 `arx_ee6d` 连续夹爪 action space。
5. `xvla/meta_arx_ee.json`
   - 保存数据根目录、三相机、episode split、25 Hz、1 秒和 domain 6。
6. 部署 adapter
   - rotation-6D 转 wxyz；
   - 连续夹爪方向恢复；
   - 30 anchor/秒到 25 control step/秒的时间重采样。

## 8. 验证标准

实现完成后必须覆盖以下测试：

1. 从指定 episode 列表中只产生被选 episode 的样本；
2. 不会跨 episode 边界读取未来动作；
3. 当前帧必须拥有完整未来 1 秒，尾部不足窗口的帧被排除；
4. 三路图像来自相同 episode、相同 frame index；
5. `image_input.shape == (3, 3, 224, 224)`；
6. `proprio.shape == (20,)`；
7. `action.shape == (30, 20)`；
8. anchor 查询时间严格为 `cur + k/30` 秒；
9. 第 30 个 anchor 对应 `cur + 1.0s`；
10. wxyz/rotation-6D 往返转换的旋转误差在容差内；
11. 连续夹爪使用 MSE，且 `preprocess()` 不清零夹爪；
12. batch 中 `domain_id` 全部为 6；
13. 训练和推理加载的 `action_mode` 都是 `arx_ee6d`；
14. 预测的 30 点轨迹能按时间正确重采样为 25 Hz 控制轨迹。

## 9. 当前最终配置摘要

```text
数据读取       新增 X-VLA DomainHandler，原生读取 LeRobot v3
Episode 选择   meta 引用固定 split JSON；Handler 按 episode ID 迭代
相机           head + left wrist + right wrist，同帧同步
原始动作       双臂 16D：xyz + wxyz + continuous gripper
模型动作       双臂 20D：xyz + rotation6d + continuous gripper
真实频率       25 Hz
预测窗口       1 秒
模型输出       30 个 anchor
Domain         6，复用 RoboTwin2/ARX-X5 domain 参数
Action space   arx_ee6d，连续夹爪 MSE
部署           30 anchors/秒重采样到25个控制点/秒，或短段执行后重规划
```
