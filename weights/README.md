# 权重说明

全部权重均为 **FP16**（从原始 FP32 转换），只用于推理，自带架构配置 `cfg`，加载时不用另外传配置。其中 78.58M 版本因超过 GitHub 单文件体积限制，按张量切分为 4 个 ~40 MiB 分片存放于目录中，`examples/chat.py` 可直接按目录路径加载。

| 路径 | 体积 | 架构 | 唯一参数 | 激活参数 | step | val_loss |
|---|---:|---|---:|---:|---:|---:|
| `m14l_d512_78m_ep2_fp16/`（4 片） | 165.7 MB | `LMP4`（14 层全独立 `d=512, i=2048`） | **78.58M** | 78.58M | 31,500 | **1.6403** |
| `m12l_d384_moe_ep3_fp16.pt` | 67.1 MB | `LMMoE`（第 3 轮，L6–L9 稀疏化） | 30.34M | 23.26M | 17,000 | 1.7744 |
| `m12l_d384_ep2_fp16.pt` | 45.8 MB | `LMP4`（第 2 轮，12 层共享 6 组） | 19.71M | 19.71M | 34,000 | 1.8439 |

> - **`m14l_d512_78m_ep2_fp16/`**：使用全量 230.9 万条语料（**已包含 `corpus/new/` 的 `C13` 数学与 `SG` 故事**，且对带资料定位语料做了 `1/3` 保留前缀 + `2/3` 剥离清洗）训满 2 轮（单卡 RTX 3070 8GB，约 6 小时），372 题盲测总分 `67.1%`（`150/199/23`）。
> - `m12l_d384_*` 两个早期权重**没有**在 `corpus/new/` 新语料上训练过。

## FP16 转换的精度损失

转换后做了同输入 logits 对照（随机 64 token）：

| | 最大绝对误差 | 平均绝对误差 | 最大概率分歧 |
|---|---:|---:|---:|
| **78.58M ep2（4 片）** | **0.0158** | **0.0019** | **0.071%** |
| MoE ep3 | 0.193 | 0.0073 | 0.29% |
| ep2 | 0.024 | 0.0032 | 0.10% |

概率分布的最大偏差均在千分之三以内，贪心解码路径一致。需要逐位复现原始结果时请用 FP32 原权重。

## 加载

### 方式一：直接运行示例脚本（同时支持单文件与 78M 分片目录）

```bash
python examples/chat.py --model weights/m14l_d512_78m_ep2_fp16 --prompt "你好，你是谁？"
python examples/chat.py --model weights/m12l_d384_moe_ep3_fp16.pt --prompt "你好，你是谁？"
```

### 方式二：Python 手动加载

```python
import json, os, torch
from model_p4 import LMP4, cfg_p4

# 1) 加载 78.58M 分片权重
base = "weights/m14l_d512_78m_ep2_fp16"
meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
sd = {}
for sh in meta["shards"]:
    sd.update(torch.load(os.path.join(base, sh), map_location="cpu", weights_only=True))
model = LMP4(cfg_p4(**meta["cfg"]))
model.load_state_dict(sd)
model.eval()

# 2) 加载 19.71M / 30.34M 单文件权重
# ck = torch.load("weights/m12l_d384_moe_ep3_fp16.pt", map_location="cpu", weights_only=False)
# from model_moe import LMMoE, cfg_moe
# model = LMMoE(cfg_moe(**ck["cfg"])); model.load_state_dict(ck["model"]); model.eval()
```

单文件 `ck`（及分片 `meta.json`）字段：`cfg`（架构配置）、`step`、`epoch`、`best_val`、`dtype`、`note`（单文件另含 `model`，分片目录另含 `shards`）。

## 量化

`moxing/quant.py` 是训练期 QAT 伪量化（int4 权重 / int8 激活，STE 直通），
按板端规格 `group=48`、`W_MAX=7`、`A_MAX=127`。它不导出文件，只做前向模拟；
要出真正的 int4 二进制需要自己写导出器。

按该规格估算的体积：78.58M ≈ 37.5 MiB，MoE ≈ 14.5 MiB，12 层 ≈ 9.4 MiB。
