# 权重说明

两个文件都是 **FP16**（从原始 FP32 转换），只用于推理。每个文件内自带 `cfg`，加载时不用另外传配置。

| 文件 | 体积 | 架构 | 唯一参数 | 激活参数 | step | val_loss |
|---|---:|---|---:|---:|---:|---:|
| `m12l_d384_moe_ep3_fp16.pt` | 67.1 MB | `LMMoE`（第 3 轮，L6–L9 稀疏化） | 30.34M | 23.26M | 17,000 | **1.7744** |
| `m12l_d384_ep2_fp16.pt` | 45.8 MB | `LMP4`（第 2 轮，12 层共享 6 组） | 19.71M | 19.71M | 34,000 | 1.8439 |

> 这两个权重**没有**在 `corpus/new/` 那三条新语料上训练过 —— 新语料是训练结束之后才产出的。
> 用全量语料重训，结果会与这里不同。

## FP16 转换的精度损失

转换后做了同输入 logits 对照（随机 64 token）：

| | 最大绝对误差 | 平均绝对误差 | 最大概率分歧 |
|---|---:|---:|---:|
| MoE ep3 | 0.193 | 0.0073 | 0.29% |
| ep2 | 0.024 | 0.0032 | 0.10% |

概率分布的最大偏差在千分之三以内，贪心解码路径一致。需要逐位复现原始结果时请用 FP32 原权重。

## 加载

```python
import torch
ck = torch.load("weights/m12l_d384_moe_ep3_fp16.pt", map_location="cpu", weights_only=False)

# MoE
from model_moe import LMMoE, cfg_moe
model = LMMoE(cfg_moe(**ck["cfg"]))
# 或 12 层版
# from model_p4 import LMP4, cfg_p4
# model = LMP4(cfg_p4(**ck["cfg"]))

model.load_state_dict(ck["model"])
model.eval()
```

`ck` 的字段：`model`（state_dict）、`cfg`（架构配置）、`step`、`epoch`、`best_val`、`dtype`、`note`。

## 量化

`moxing/quant.py` 是训练期 QAT 伪量化（int4 权重 / int8 激活，STE 直通），
按板端规格 `group=48`、`W_MAX=7`、`A_MAX=127`。它不导出文件，只做前向模拟；
要出真正的 int4 二进制需要自己写导出器。

按该规格估算的体积：MoE ≈ 14.5 MiB，12 层 ≈ 9.4 MiB。
