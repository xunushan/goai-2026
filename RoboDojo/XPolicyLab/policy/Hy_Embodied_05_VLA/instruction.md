# Hy-VLA 独立服务

## 1. 安装环境

```bash
cd RoboDojo/XPolicyLab/policy/Hy_Embodied_05_VLA
bash install.sh

# Hy-VLA 使用 uv 创建的项目虚拟环境，不使用 Conda。
source Hy-Embodied-0.5-VLA/.venv/bin/activate
```

## 2. 下载模型

```bash
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "ckpt/RoboDojo/hy_vla/zzilch/rd20/*" \
  --local-dir /data/checkpoints/hy_vla
```

模型目录：

```text
/data/checkpoints/hy_vla/ckpt/RoboDojo/hy_vla/zzilch/rd20
```

## 3. 启动服务

```bash
cd RoboDojo/XPolicyLab/policy/Hy_Embodied_05_VLA
source Hy-Embodied-0.5-VLA/.venv/bin/activate

bash serve_remote.sh \
  /data/checkpoints/hy_vla/ckpt/RoboDojo/hy_vla/zzilch/rd20 \
  stack_blocks 0 80 0.0.0.0
```

参数依次为：模型目录、任务名、GPU、端口、监听地址。

`serve_remote.sh` 也会检查并激活上述 `.venv`。仿真机按照 [`docs/仿真测试指南.md`](../../../../docs/仿真测试指南.md) 连接服务机 IP 和端口 `80`。评测动作类型使用 `ee`。
