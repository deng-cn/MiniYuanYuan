# -*- coding: utf-8 -*-
"""QAT 量化模块 —— 与板端逐项一致

板端规格（烧录端 export_v4r10.py / 固件）：
  · 权重：int4 对称，按输入维分组 group=72，scale = max(|w|)/7，clip(round(w/scale), -7, 7)
  · 激活：int8 对称，group=72，scale = max(|x|)/127，clip(-127, 127)
  · 不做量化：所有 RMSNorm、MoE router、kv_delta 的 up 层、位置编码、MTP 头
  · 量化对象：q/k/v/o_proj、gate/up/down_proj、kv_delta.down、embed_tokens / lm_head（共享一份）

训练时用 STE（直通估计）：
  forward 走量化值，backward 把梯度原样传给 fp32 master 权重，
  因此参数本身始终是 fp32，导出时再按同一公式量化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

W_MAX = 7        # int4 对称上限（clamp -7..7）
A_MAX = 127      # int8 对称上限
GROUP = 48       # 1125 默认分组（48 整除 d=480 与 2d=960，无 padding 浪费）
W_CLIP = 0.90    # 权重截断系数 a：scale = amax * a / 7（实测全量最优，导出端须同值）

# 需要量化的 Linear 叶子名（注意：self_attn.gate_proj 为动态路由门控，严禁量化）
_LINEAR_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj", "lm_head")


def _group_quant(x, group, qmax, clip=1.0):
    """沿最后一维按 group 分组做对称量化，返回 fp32 反量化张量。"""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    n = flat.shape[-1]
    pad = (-n) % group
    if pad:
        flat = F.pad(flat, (0, pad))
    g = flat.reshape(-1, group)
    scale = (g.abs().amax(dim=-1, keepdim=True) * clip / qmax).clamp_min(1e-8)
    q = (g / scale).round().clamp_(-qmax, qmax)
    out = (q * scale).reshape(flat.shape)
    if pad:
        out = out[:, :n]
    return out.reshape(shape)


def _ste(x, xq):
    """直通估计：前向取量化值，反向梯度等于 1。"""
    return x + (xq.to(x.dtype) - x).detach()


def quant_weight(w, group=GROUP, clip=W_CLIP):
    """int4 伪量化权重（STE），返回与 w 同 dtype。"""
    return _ste(w, _group_quant(w.float(), group, W_MAX, clip))


def quant_act(x, group=GROUP):
    """int8 伪量化激活（STE），返回与 x 同 dtype（激活不带截断系数）。"""
    return _ste(x, _group_quant(x.float(), group, A_MAX, 1.0))


class QLinear(nn.Module):
    """量化 Linear：输入走 int8，权重走 int4，bias 不量化。"""

    def __init__(self, base: nn.Linear, group: int = GROUP):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.group = group
        self.weight = base.weight          # 共享原 Parameter，优化器引用不变
        self.bias = base.bias

    def forward(self, x):
        return F.linear(quant_act(x, self.group),
                        quant_weight(self.weight, self.group), self.bias)

    def extra_repr(self):
        return "in=%d, out=%d, group=%d, w=int4, a=int8" % (
            self.in_features, self.out_features, self.group)


class QEmbedding(nn.Module):
    """量化词嵌入：只量化权重（token id 输入不需要激活量化）。"""

    def __init__(self, base: nn.Embedding, group: int = GROUP):
        super().__init__()
        self.num_embeddings = base.num_embeddings
        self.embedding_dim = base.embedding_dim
        self.padding_idx = base.padding_idx
        self.group = group
        self.weight = base.weight

    def forward(self, ids):
        return F.embedding(ids, quant_weight(self.weight, self.group), self.padding_idx)


def _parent_of(model, name):
    if "." not in name:
        return model, name
    parent = model.get_submodule(name.rsplit(".", 1)[0])
    return parent, name.rsplit(".", 1)[-1]


def apply_qat(model, group: int = GROUP):
    """把目标 Linear / Embedding 替换成量化版本（原地修改）。"""
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        # 明确排除注意力动态门控（self_attn.gate_proj 是 router，不做量化）
        if name.endswith(".self_attn.gate_proj"):
            continue
        is_bicross = (".mlp." in name or ".bicross." in name or ".ffn_groups." in name or ".ffn." in name) and (leaf in ("q", "down"))
        is_kv_down = name.endswith(".down") and ".kv_deltas." in name
        if leaf in _LINEAR_LEAVES or is_bicross or is_kv_down:
            targets.append((name, mod, "linear"))
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            targets.append((name, mod, "embed"))

    n_lin = n_emb = 0
    for name, mod, kind in targets:
        parent, leaf = _parent_of(model, name)
        if kind == "linear":
            setattr(parent, leaf, QLinear(mod, group))
            n_lin += 1
        else:
            setattr(parent, leaf, QEmbedding(mod, group))
            n_emb += 1

    tied = None
    try:
        tied = model.lm_head.weight is model.model.embed_tokens.weight
    except AttributeError:
        pass
    return {"linear": n_lin, "embedding": n_emb, "group": group,
            "w": "int4/group%d/a%.2f" % (group, W_CLIP),
            "a": "int8/group%d" % group, "tied_embed": tied}
