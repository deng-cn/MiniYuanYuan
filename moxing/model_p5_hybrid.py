# -*- coding: utf-8 -*-
"""model_p5_hybrid.py —— 5 号选手（A 方案：减维增深 SwiGLU + BiCrossGate 交替混合架构）

核心设计（d=192, 16 层, 6 Q 头 / 2 KV 头, hd=32, 总参 ~12.02M）：
  · 偶数层（0, 2, ..., 14）：标准 SwiGLU（i=960, 5.00×），负责稳固知识底座与词表分布、抑制复读；
  · 奇数层（1, 3, ..., 15）：BiCrossGateFFN（i=624, 3.25×），负责二阶跨维特征交叉与上下文定位/推理；
  · 注意力：统一采用 StandardGQAAttention（QK-Norm + RoPE + 原生 GQA cuDNN SDPA）。
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _ckpt

from model_1126 import (
    RMSNorm,
    SwiGLUFFN,
    precompute_freqs_cis,
    _MtpHead,
)
from model_arena import StandardGQAAttention, ArenaBackbone
from model_p4 import BiCrossGateFFN, LMP4, cfg_p4


def _is_bicross_layer(idx, cfg):
    pattern = str(cfg.get("hybrid_pattern") or "alt_even_swiglu").lower()
    if pattern == "alt_even_bicross":
        return (idx % 2) == 0
    return (idx % 2) == 1


class P5HybridBlock(nn.Module):
    def __init__(self, cfg, idx=0):
        super().__init__()
        d = int(cfg["hidden_size"])
        eps = float(cfg["rms_norm_eps"])
        self.idx = idx
        self.is_bicross = _is_bicross_layer(idx, cfg)
        self.input_layernorm = RMSNorm(d, eps=eps)
        self.self_attn = StandardGQAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        sub_cfg = dict(cfg)
        if self.is_bicross:
            sub_cfg["intermediate_size"] = int(cfg["bicross_intermediate_size"])
            self.ffn_inter = sub_cfg["intermediate_size"]
            self.ffn = BiCrossGateFFN(sub_cfg)
        else:
            sub_cfg["intermediate_size"] = int(cfg["swiglu_intermediate_size"])
            self.ffn_inter = sub_cfg["intermediate_size"]
            self.ffn = SwiGLUFFN(sub_cfg)

    def forward(self, hidden_states, q_rope, k_rope, past_kv=None, use_cache=False, attention_mask=None):
        x_norm = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            x_norm, q_rope, k_rope, past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ffn(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class P5HybridBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers = int(cfg["num_hidden_layers"])
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        d = int(cfg["hidden_size"])
        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), d)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.layers = nn.ModuleList([P5HybridBlock(cfg, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(d, eps=float(cfg["rms_norm_eps"]))
        cos, sin = precompute_freqs_cis(
            int(cfg["head_dim"]), int(cfg["max_position_embeddings"]),
            rope_base=float(cfg["rope_theta"]), rope_scaling=cfg.get("rope_scaling"))
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, position_offset=None):
        seq_len = input_ids.shape[1]
        if past_key_values is None:
            past = [None] * self.n_layers
            past_len = 0
        else:
            past = list(past_key_values)
            past_len = ArenaBackbone._past_len(past[0])
        if position_offset is None:
            position_offset = past_len
        kv_len = past_len + seq_len
        k_start = position_offset - past_len
        if k_start < 0 or position_offset + seq_len > self.freqs_cos.shape[0]:
            raise ValueError("序列位置越界")
        q_rope = (self.freqs_cos[position_offset:position_offset + seq_len],
                  self.freqs_sin[position_offset:position_offset + seq_len])
        k_rope = (self.freqs_cos[k_start:k_start + kv_len],
                  self.freqs_sin[k_start:k_start + kv_len])

        hidden = self.dropout(self.embed_tokens(input_ids))
        if self.grad_checkpoint and self.training and not use_cache and past_key_values is None:
            for layer in self.layers:
                def _fn(h, _lay=layer):
                    out_h, _ = _lay(h, q_rope, k_rope, None, False, attention_mask)
                    return out_h
                hidden = _ckpt(_fn, hidden, use_reentrant=False)
            return self.norm(hidden), []

        presents = []
        for layer, pkv in zip(self.layers, past):
            hidden, pres = layer(hidden, q_rope, k_rope, past_kv=pkv,
                                 use_cache=use_cache, attention_mask=attention_mask)
            if use_cache:
                presents.append(pres)
        return self.norm(hidden), presents


class LMP5Hybrid(LMP4):
    """5 号选手语言模型（减维增深 SwiGLU + BiCrossGate 交替混合）。"""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        cfg = cfg_p5_hybrid(**cfg)
        self.cfg = cfg
        self.model = P5HybridBackbone(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        n_mtp = max(0, int(cfg.get("mtp") or 0) - 1)
        self.mtp_heads = nn.ModuleList([_MtpHead(cfg) for _ in range(n_mtp)])

        self.apply(self._init_weights)
        if cfg.get("resid_init_scale", True):
            r_std = 0.02 / math.sqrt(2.0 * cfg["num_hidden_layers"])
            for layer in self.model.layers:
                nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=r_std)
                ffn_std = r_std / math.sqrt(float(layer.ffn_inter) / float(cfg["hidden_size"]))
                nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=ffn_std)


def cfg_p5_hybrid(**kw):
    kw.pop("arch_type", None)
    d = int(kw.pop("hidden_size", 192))
    attn_heads = int(kw.pop("num_attention_heads", 6))
    inter_fallback = kw.pop("intermediate_size", 960)
    swiglu_i = int(kw.pop("swiglu_intermediate_size", inter_fallback))
    bicross_i = int(kw.pop("bicross_intermediate_size", 624))
    cfg = cfg_p4(
        arch_type="std_gqa_hybrid_bicross",
        hidden_size=d,
        num_hidden_layers=int(kw.pop("num_hidden_layers", 16)),
        num_attention_heads=attn_heads,
        num_key_value_heads=int(kw.pop("num_key_value_heads", 2)),
        head_dim=int(kw.pop("head_dim", d // attn_heads)),
        intermediate_size=swiglu_i,
        swiglu_intermediate_size=swiglu_i,
        bicross_intermediate_size=bicross_i,
        hybrid_pattern=str(kw.pop("hybrid_pattern", "alt_even_swiglu")).lower(),
        **kw,
    )
    return cfg


def param_report_p5_hybrid(model=None, cfg=None, verbose=True):
    cfg = dict(cfg or (model.cfg if model is not None else {}))
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    h_q, h_kv, hd = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
    sw_i = int(cfg["swiglu_intermediate_size"])
    bc_i = int(cfg["bicross_intermediate_size"])
    n_bc = sum(1 for i in range(L) if _is_bicross_layer(i, cfg))
    n_sw = L - n_bc
    embed = int(cfg["vocab_size"]) * d * (1 if cfg.get("tie_word_embeddings", True) else 2)
    attn_per = d * (h_q * hd) + 2 * d * (h_kv * hd) + (h_q * hd) * d + 2 * hd
    sw_per = 3 * d * sw_i
    bc_per = 2 * d * d + 4 * d * bc_i
    norms_per = 2 * d
    total = embed + L * (attn_per + norms_per) + n_sw * sw_per + n_bc * bc_per + d
    measured = model.num_params() if model is not None else None
    if verbose:
        print("=" * 82)
        print("5号选手(A方案) 减维增深交替混合  d=%d ｜ %d层 (%d×SwiGLU i=%d + %d×BiCross i=%d)"
              % (d, L, n_sw, sw_i, n_bc, bc_i))
        print("  占比：Embed %.2fM(%.1f%%) ｜ Attn %.2fM(%.1f%%) ｜ FFN %.2fM(%.1f%%)"
              % (embed / 1e6, 100.0 * embed / total,
                 (L * attn_per) / 1e6, 100.0 * (L * attn_per) / total,
                 (n_sw * sw_per + n_bc * bc_per) / 1e6,
                 100.0 * (n_sw * sw_per + n_bc * bc_per) / total))
        print("  公式总数 %s (%.4f M) ｜ 4bit ≈ %.2f MiB ｜ bf16 ≈ %.2f MiB"
              % ("{:,}".format(total), total / 1e6, total * 0.5 / 1048576, total * 2.0 / 1048576))
        if measured is not None:
            print("  torch 实测 %s (%.4f M) ｜ 差异 %d"
                  % ("{:,}".format(measured), measured / 1e6, measured - total))
            assert measured == total, "公式 %d != 实测 %d" % (total, measured)
        print("=" * 82)
    return measured, total
