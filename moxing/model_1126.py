# -*- coding: utf-8 -*-
"""model_1126.py —— 1126 架构（方案 A）：10 层独立 × 4 套并行 Q × 无 W_K/W_V × BiCross + SwiGLU(2d)

核心规格（2026-11-25 拍板）：
    1. 基础维度：hidden_size = 384 ｜ num_hidden_layers = 10（全独立，无权重/FFN 共享）
       最新词表：vocab_v2 = 8,302（tie_word_embeddings=True）
    2. 单层三段式串联（方案 A，KV 保持 d=384 维）：
       (1) Pre-Norm -> 4 套并行 Q 无 KV 投影注意力（48 Q 头查 12 KV 头, hd=32 满速 cuDNN）
           + 48 路逐头动态门控（零初始化）+ 共享 W_O (384->384) -> 残差
       (2) Pre-Norm -> BiCross 双线性交叉层（QBilinearFFN, 4d²）-> 残差
       (3) Pre-Norm -> SwiGLU 2d 扩维 FFN（384->768->384, 6d²）-> 残差
       下一层的 4 套 Q 直接把本层「SwiGLU 2d 后的最终输出」经 Pre-Norm 作为 K 和 V（不再查 BiCross 刚出来的状态）。
    3. 彻底取消 W_K 与 W_V：
       每层推理 KV Cache 仅保存 1 份未旋转的干净状态 [B, T, 12, 32]；
       V 零污染；K 仅在点积前临时过逐头 k_norm + 原生半维 RoPE，绝不写回缓存。
    4. 集成 1125 独立审查全部优化：
       - PyTorch 原生融合 F.rms_norm + 逐头 MultiHeadRMSNorm（释放 4 套 Q 注意力温度分化）
       - 半维原生 dtype 高速 RoPE（零 fp32 临时拷贝）
       - 三残差分支均衡缩放初始化（r_std = 0.02/sqrt(3L)，BiCross.down 与 SwiGLU.down_proj 再除 sqrt(2)）
       - position_offset/k_start 偏移对齐、全 -100 防 NaN、logits_to_keep=1 加速解码、grad_checkpoint 支持

本文件自包含，仅依赖 torch。
"""
import contextlib
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as _ckpt

IGNORE = -100

try:
    from torch.nn.attention import SDPBackend as _SDPB, sdpa_kernel as _sdpa_kernel
    _BACKENDS = {
        "flash": _SDPB.FLASH_ATTENTION,
        "efficient": _SDPB.EFFICIENT_ATTENTION,
        "math": _SDPB.MATH,
        "cudnn": _SDPB.CUDNN_ATTENTION,
    }
except Exception:
    _BACKENDS = {}
    _sdpa_kernel = None


def _probe_sdpa_gqa():
    try:
        q = torch.zeros(1, 2, 1, 8)
        k = torch.zeros(1, 1, 1, 8)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_HAS_GQA = _probe_sdpa_gqa()


def sdpa_ctx(backend="cudnn", dtype=None):
    """返回 SDPA backend 上下文；cudnn 优先并附 efficient+math 兜底，fp32 自动回退。"""
    if not backend or backend == "auto" or _sdpa_kernel is None or backend not in _BACKENDS:
        return contextlib.nullcontext()
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
        if backend in ("cudnn", "flash"):
            backend = "efficient"
        return _sdpa_kernel([_BACKENDS[backend], _BACKENDS["math"]])
    if backend == "cudnn":
        return _sdpa_kernel([_BACKENDS["cudnn"], _BACKENDS["efficient"], _BACKENDS["math"]])
    return _sdpa_kernel([_BACKENDS[backend]])


# ======================================================================
# 基础件
# ======================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class MultiHeadRMSNorm(nn.Module):
    """逐头独立尺度的头维 RMSNorm：归一化走融合 F.rms_norm，再乘逐头独立权重。"""

    def __init__(self, weight_shape, eps=1e-6):
        super().__init__()
        self.head_dim = int(weight_shape[-1])
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(*weight_shape))

    def forward(self, x):
        normed = F.rms_norm(x, (self.head_dim,), None, self.eps)
        return normed * self.weight.to(normed.dtype)


def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low)
                               / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return cos, sin


def apply_rope(x, cos, sin):
    """通用高速 RoPE：支持 4D [B, T, H, D] 与 5D [B, T, M, H, D]，半维原生 dtype 旋转。"""
    t, d = cos.shape
    half = d // 2
    view_shape = (1, t) + (1,) * (x.ndim - 3) + (half,)
    c = cos[:, :half].to(x.dtype).view(view_shape)
    s = sin[:, :half].to(x.dtype).view(view_shape)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def repeat_kv_heads(x, n_rep):
    if n_rep == 1:
        return x
    b, h_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h_kv, n_rep, t, d).reshape(b, h_kv * n_rep, t, d)


# ======================================================================
# 4 套并行 Q + 无 W_K/W_V + 逐头动态门控融合 Attention
# ======================================================================
class ParallelQNoKVAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d = int(cfg["hidden_size"])
        self.n_kv_heads = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg["head_dim"])
        self.n_q_sets = int(cfg["num_q_sets"])
        self.n_q_heads = self.n_kv_heads * self.n_q_sets
        self.gate_mode = str(cfg.get("gate_mode", "head")).lower()
        self.use_o_proj = bool(cfg.get("use_o_proj", True))
        self.use_k_norm = bool(cfg.get("use_k_norm", True))
        self.per_head_qk_norm = bool(cfg.get("per_head_qk_norm", True))
        self.use_native_gqa = bool(cfg.get("use_native_gqa", True)) and _SDPA_HAS_GQA
        self.attn_backend = str(cfg.get("attn_backend", "cudnn"))
        self.dropout = float(cfg.get("dropout", 0.0))

        self.q_proj = nn.Linear(self.d, self.n_q_sets * self.d, bias=False)
        if self.per_head_qk_norm:
            self.q_norm = MultiHeadRMSNorm((self.n_q_sets, self.n_kv_heads, self.head_dim), eps=cfg["rms_norm_eps"])
        else:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        if self.use_k_norm:
            if self.per_head_qk_norm:
                self.k_norm = MultiHeadRMSNorm((self.n_kv_heads, self.head_dim), eps=cfg["rms_norm_eps"])
            else:
                self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        else:
            self.k_norm = None

        gate_out = (self.n_kv_heads * self.n_q_sets) if self.gate_mode == "head" else self.n_q_sets
        self.gate_proj = nn.Linear(self.d, gate_out, bias=True)
        self.o_proj = nn.Linear(self.d, self.d, bias=False) if self.use_o_proj else None
        self.resid_dropout = nn.Dropout(self.dropout)
        self._gate_stats = None

    def reset_gate(self):
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def enable_gate_stats(self):
        self._gate_stats = {"probs": None, "entropy": 0.0, "pair_cos": None, "calls": 0}

    def pop_gate_stats(self):
        s, self._gate_stats = self._gate_stats, None
        if not s or s["calls"] == 0:
            return None
        n = float(s["calls"])
        return {
            "mean_q_probs": (s["probs"] / n).tolist(),
            "norm_entropy": s["entropy"] / n,
            "pairwise_cos": (s["pair_cos"] / n).tolist(),
        }

    @torch.no_grad()
    def _accum_gate_stats(self, gate_w, attn_out):
        p = gate_w.detach().float()
        mean_p = p.mean(dim=(0, 1, 2)).cpu()
        ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1).mean().item() / math.log(self.n_q_sets)
        y = attn_out.detach().float().permute(0, 1, 3, 2, 4).reshape(-1, self.n_q_sets, self.d)
        u = F.normalize(y, dim=-1)
        cos = torch.einsum("nmd,nkd->mk", u, u).cpu() / float(u.shape[0])
        if self._gate_stats["probs"] is None:
            self._gate_stats["probs"] = mean_p
            self._gate_stats["pair_cos"] = cos
        else:
            self._gate_stats["probs"] += mean_p
            self._gate_stats["pair_cos"] += cos
        self._gate_stats["entropy"] += ent
        self._gate_stats["calls"] += 1

    @staticmethod
    def _causal_mask(q_len, kv_len, device, dtype):
        off = kv_len - q_len
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=off)
        m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        return m.masked_fill(~causal, torch.finfo(dtype).min)

    def forward(self, x_norm, q_rope, k_rope, past_kv=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x_norm.shape

        # 1. 无 W_K/W_V：直接将本层归一化后的状态（来自上一层 SwiGLU 2d 最终输出）作为单份干净 KV
        kv_raw = x_norm.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        kv_full = torch.cat((past_kv, kv_raw), dim=1) if past_kv is not None else kv_raw
        present = kv_full if use_cache else None
        kv_len = kv_full.shape[1]

        # 2. V 保持未旋转；K 临时过 k_norm + RoPE（绝不写回缓存）
        xv = kv_full.transpose(1, 2)
        xk_base = self.k_norm(kv_full) if self.k_norm is not None else kv_full
        xk = apply_rope(xk_base, k_rope[0], k_rope[1]).transpose(1, 2)

        # 3. 4 套并行 Q 打包投影 + GQA 正确头排列 [B, H_kv, M, T, hd] -> [B, 48, T, hd]
        xq = self.q_proj(x_norm).view(bsz, seq_len, self.n_q_sets, self.n_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = apply_rope(xq, q_rope[0], q_rope[1])
        xq = xq.permute(0, 3, 2, 1, 4).reshape(bsz, self.n_q_heads, seq_len, self.head_dim)

        # 4. 原生 GQA SDPA 一次算完
        p = self.dropout if self.training else 0.0
        single_decode_no_mask = (seq_len == 1) and (attention_mask is None)
        needs_mask = (not single_decode_no_mask) and ((kv_len != seq_len) or (attention_mask is not None))
        with sdpa_ctx(self.attn_backend, xq.dtype):
            if self.use_native_gqa:
                if single_decode_no_mask:
                    out = F.scaled_dot_product_attention(
                        xq, xk, xv, dropout_p=p, is_causal=False, enable_gqa=True)
                elif not needs_mask:
                    out = F.scaled_dot_product_attention(
                        xq, xk, xv, dropout_p=p, is_causal=True, enable_gqa=True)
                else:
                    m = self._causal_mask(seq_len, kv_len, xq.device, xq.dtype)
                    if attention_mask is not None:
                        m = m.masked_fill(attention_mask[:, None, None, :].to(xq.device) == 0,
                                          torch.finfo(xq.dtype).min)
                    out = F.scaled_dot_product_attention(
                        xq, xk, xv, attn_mask=m, dropout_p=p, enable_gqa=True)
            else:
                xk_rep = repeat_kv_heads(xk, self.n_q_sets)
                xv_rep = repeat_kv_heads(xv, self.n_q_sets)
                if single_decode_no_mask:
                    out = F.scaled_dot_product_attention(xq, xk_rep, xv_rep, dropout_p=p, is_causal=False)
                elif not needs_mask:
                    out = F.scaled_dot_product_attention(xq, xk_rep, xv_rep, dropout_p=p, is_causal=True)
                else:
                    m = self._causal_mask(seq_len, kv_len, xq.device, xq.dtype)
                    if attention_mask is not None:
                        m = m.masked_fill(attention_mask[:, None, None, :].to(xq.device) == 0,
                                          torch.finfo(xq.dtype).min)
                    out = F.scaled_dot_product_attention(xq, xk_rep, xv_rep, attn_mask=m, dropout_p=p)

        # 5. 动态门控加权融合 + 共享 W_O
        attn_out = out.view(bsz, self.n_kv_heads, self.n_q_sets, seq_len, self.head_dim)
        attn_out = attn_out.permute(0, 3, 1, 2, 4)  # [B, T, H_kv, M, hd]

        gate_logits = self.gate_proj(x_norm)
        if self.gate_mode == "head":
            gate_logits = gate_logits.view(bsz, seq_len, self.n_kv_heads, self.n_q_sets)
        else:
            gate_logits = gate_logits.view(bsz, seq_len, 1, self.n_q_sets)
        gate_w = F.softmax(gate_logits.float(), dim=-1).to(attn_out.dtype)

        if self._gate_stats is not None:
            self._accum_gate_stats(gate_w, attn_out)

        fused = (attn_out * gate_w.unsqueeze(-1)).sum(dim=3).reshape(bsz, seq_len, self.d)
        if self.o_proj is not None:
            fused = self.o_proj(fused)
        return self.resid_dropout(fused), present


# ======================================================================
# 双串联 FFN：BiCross (QBilinearFFN) -> SwiGLU 2d
# ======================================================================
class QBilinearFFN(nn.Module):
    """BiCross 双分支二次门控（4d² 参数/层，零冗余高维张量）。"""

    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["hidden_size"])
        self.act1 = str(cfg.get("qbilinear_act1") or "gelu").lower()
        self.act2 = str(cfg.get("qbilinear_act2") or "silu").lower()
        self.q = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    @staticmethod
    def _act(name, x):
        if name == "gelu":
            return F.gelu(x)
        if name == "silu":
            return F.silu(x)
        if name == "tanh":
            return torch.tanh(x)
        return torch.sigmoid(x)

    def forward(self, x):
        q1, q2 = self.q(x).chunk(2, dim=-1)
        a = self._act(self.act1, q1 * x)
        c = self._act(self.act2, q2 * x)
        return self.down(torch.cat((a, c), dim=-1))


class SwiGLUFFN(nn.Module):
    """扩大两倍（2d）SwiGLU FFN（6d² 参数/层，384 -> 768 -> 384）。"""

    def __init__(self, cfg):
        super().__init__()
        h, i = int(cfg["hidden_size"]), int(cfg["intermediate_size"])
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ParallelQDualFFNBlock(nn.Module):
    """1126 三段式独立层：
      1) h = h + 4Q_NoKV_Attn(input_layernorm(h))
      2) h = h + BiCross(post_attention_layernorm(h))
      3) h = h + SwiGLU_2d(post_bicross_layernorm(h))
    最终输出的 h（经 SwiGLU 2d 后）直接作为下一层 4 套 Q 的 K/V 来源。
    """

    def __init__(self, cfg, idx=0):
        super().__init__()
        d = int(cfg["hidden_size"])
        eps = float(cfg["rms_norm_eps"])
        self.idx = idx
        self.input_layernorm = RMSNorm(d, eps=eps)
        self.self_attn = ParallelQNoKVAttention(cfg)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.bicross = QBilinearFFN(cfg)
        self.post_bicross_layernorm = RMSNorm(d, eps=eps)
        self.swiglu = SwiGLUFFN(cfg)

    def forward(self, hidden_states, q_rope, k_rope, past_kv=None, use_cache=False, attention_mask=None):
        x_norm = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            x_norm, q_rope, k_rope, past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.bicross(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + self.swiglu(self.post_bicross_layernorm(hidden_states))
        return hidden_states, present


class _MtpHead(nn.Module):
    """可选并行多 token 预测辅助头（1126 默认关闭 mtp=0，保留接口兼容）。"""

    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["hidden_size"])
        self.norm = RMSNorm(d, eps=float(cfg["rms_norm_eps"]))
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.proj(self.norm(x))


# ======================================================================
# 1126 主干与 LM 封装
# ======================================================================
class ParallelQDualFFNBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers = int(cfg["num_hidden_layers"])
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        d = int(cfg["hidden_size"])

        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), d)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.layers = nn.ModuleList([ParallelQDualFFNBlock(cfg, i) for i in range(self.n_layers)])
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
            if len(past) != self.n_layers:
                raise ValueError("past_key_values 长度 %d，应为 %d" % (len(past), self.n_layers))
            past_len = past[0].shape[1] if past[0] is not None else 0

        if position_offset is None:
            position_offset = past_len
        kv_len = past_len + seq_len
        k_start = position_offset - past_len
        if k_start < 0 or position_offset + seq_len > self.freqs_cos.shape[0] or k_start + kv_len > self.freqs_cos.shape[0]:
            raise ValueError("序列位置 k_start=%d / q_end=%d / k_end=%d 超过 max_position_embeddings %d"
                             % (k_start, position_offset + seq_len, k_start + kv_len, self.freqs_cos.shape[0]))

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


class LM1126(nn.Module):
    """1126 主语言模型（10 层独立 4Q-NoKV + BiCross + SwiGLU 2d）。"""

    def __init__(self, cfg):
        super().__init__()
        cfg = cfg_1126(**cfg)
        self.cfg = cfg
        self.model = ParallelQDualFFNBackbone(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        n_mtp = max(0, int(cfg.get("mtp") or 0) - 1)
        self.mtp_heads = nn.ModuleList([_MtpHead(cfg) for _ in range(n_mtp)])

        self.apply(self._init_weights)
        # 1) 三残差分支缩放初始化（每层 3 次残差相加 -> sqrt(3L)；2d 宽 down 再除 sqrt(2) 对齐 fan-in）
        if cfg.get("resid_init_scale", True):
            r_std = 0.02 / math.sqrt(3.0 * cfg["num_hidden_layers"])
            ffn_std = r_std / math.sqrt(2.0)
            for layer in self.model.layers:
                if layer.self_attn.o_proj is not None:
                    nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=r_std)
                nn.init.normal_(layer.bicross.down.weight, mean=0.0, std=ffn_std)
                nn.init.normal_(layer.swiglu.down_proj.weight, mean=0.0, std=ffn_std)
        # 2) 动态门控归零，确保起点严格均分 0.25×4
        for layer in self.model.layers:
            layer.self_attn.reset_gate()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_grad_checkpoint(self, enabled=True):
        flag = bool(enabled) and (enabled != "none")
        self.model.grad_checkpoint = flag
        self.cfg["grad_checkpoint"] = flag

    def enable_gate_stats(self):
        for layer in self.model.layers:
            layer.self_attn.enable_gate_stats()

    def pop_gate_stats(self):
        return [layer.self_attn.pop_gate_stats() for layer in self.model.layers]

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kw):
        hidden, presents = self.model(
            input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, position_offset=kw.get("position_offset"))
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep else slice(None)
        logits = self.lm_head(hidden[:, sl, :])
        loss = None
        mtp_loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            if (y != IGNORE).any():
                loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=IGNORE)
            else:
                loss = logits.sum() * 0.0
            if len(self.mtp_heads) > 0 and not logits_to_keep:
                weights = self.cfg.get("mtp_weights") or [0.2, 0.1]
                mtp_loss = 0.0
                for step, head in enumerate(self.mtp_heads, start=2):
                    lg = self.lm_head(head(hidden))
                    if lg.shape[1] <= step:
                        continue
                    a = lg[:, :-step, :].contiguous()
                    b = labels[:, step:].contiguous()
                    if not (b != IGNORE).any():
                        continue
                    l = F.cross_entropy(a.view(-1, a.size(-1)), b.view(-1), ignore_index=IGNORE)
                    w = weights[step - 2] if step - 2 < len(weights) else 0.1
                    mtp_loss = mtp_loss + w * l
                    loss = loss + w * l
        return {"loss": loss, "logits": logits, "past_key_values": presents,
                "hidden_states": hidden, "mtp_loss": mtp_loss}

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=128, temperature=0.85,
                 top_p=0.9, top_k=50, eos_token_id=None, repetition_penalty=1.0,
                 no_repeat_ngram_size=0, do_sample=True, use_cache=True, past_key_values=None,
                 context_len=0):
        eos_token_id = self.cfg.get("eos_token_id", 3) if eos_token_id is None else eos_token_id
        if isinstance(eos_token_id, (list, tuple, set)):
            stop_ids = set(int(x) for x in eos_token_id)
        else:
            stop_ids = {int(eos_token_id)}
        ids = input_ids
        if not use_cache:
            past_key_values = None
        use_attn_mask = attention_mask is not None
        max_pos = int(self.cfg["max_position_embeddings"])
        generated = []
        for _ in range(max_new_tokens):
            if ids.shape[1] >= max_pos:
                break
            past_len = past_key_values[0].shape[1] if (use_cache and past_key_values) else 0
            out = self.forward(ids[:, past_len:], attention_mask if use_attn_mask else None,
                               past_key_values, use_cache=use_cache, logits_to_keep=1,
                               position_offset=past_len if use_cache else 0)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            c0 = int(context_len) if context_len and context_len > 0 else 0
            if repetition_penalty != 1.0:
                for i in range(ids.shape[0]):
                    seen = torch.unique(ids[i, c0:] if c0 else ids[i])
                    sc = logits[i, seen]
                    logits[i, seen] = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
            if no_repeat_ngram_size and no_repeat_ngram_size >= 2 and ids.shape[1] >= no_repeat_ngram_size:
                n = int(no_repeat_ngram_size)
                L = ids.shape[1]
                win = ids.unfold(1, n - 1, 1)
                pref = ids[:, -(n - 1):].unsqueeze(1)
                eq = (win == pref).all(-1)[:, :L - n + 1]
                if c0:
                    eq[:, :c0] = False
                ban = torch.zeros_like(logits, dtype=torch.bool)
                ban.scatter_(1, ids[:, n - 1:L], eq)
                logits = logits.masked_fill(ban, float("-inf"))
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and top_p < 1.0:
                srt, idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1) > top_p
                cum[..., 1:] = cum[..., :-1].clone()
                cum[..., 0] = False
                logits = logits.masked_fill(cum.scatter(1, idx, cum), float("-inf"))
            nxt = (torch.multinomial(torch.softmax(logits, dim=-1), 1) if do_sample
                   else torch.argmax(logits, dim=-1, keepdim=True))
            ids = torch.cat([ids, nxt], dim=-1)
            if use_attn_mask:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
            past_key_values = out["past_key_values"] if use_cache else None
            generated.append(int(nxt[0, 0]))
            if int(nxt[0, 0]) in stop_ids:
                break
        return {"sequences": ids, "new_tokens": generated, "past_key_values": past_key_values}

    def num_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


# ======================================================================
# 配置 / 校验 / 参数账本 / KV Cache 估算
# ======================================================================
def cfg_1126(**kw):
    d = int(kw.pop("hidden_size", 384))
    kv_heads = int(kw.pop("num_key_value_heads", 12))
    q_sets = int(kw.pop("num_q_sets", 4))
    cfg = {
        "vocab_size": int(kw.pop("vocab_size", 8302)),
        "hidden_size": d,
        "num_hidden_layers": int(kw.pop("num_hidden_layers", 10)),
        "num_key_value_heads": kv_heads,
        "num_q_sets": q_sets,
        "num_attention_heads": int(kw.pop("num_attention_heads", kv_heads * q_sets)),
        "head_dim": int(kw.pop("head_dim", d // kv_heads)),
        "gate_mode": str(kw.pop("gate_mode", "head")).lower(),
        "use_o_proj": bool(kw.pop("use_o_proj", True)),
        "use_k_norm": bool(kw.pop("use_k_norm", True)),
        "per_head_qk_norm": bool(kw.pop("per_head_qk_norm", True)),
        "use_native_gqa": bool(kw.pop("use_native_gqa", True)),
        "resid_init_scale": bool(kw.pop("resid_init_scale", True)),
        "qbilinear_act1": str(kw.pop("qbilinear_act1", "gelu")).lower(),
        "qbilinear_act2": str(kw.pop("qbilinear_act2", "silu")).lower(),
        "intermediate_size": int(kw.pop("intermediate_size", d * 2)),
        "mtp": int(kw.pop("mtp", 0) or 0),
        "mtp_weights": list(kw.pop("mtp_weights", [0.2, 0.1]) or []),
        "max_position_embeddings": int(kw.pop("max_position_embeddings", 1088)),
        "rope_theta": float(kw.pop("rope_theta", 1e6)),
        "rope_scaling": kw.pop("rope_scaling", None),
        "rms_norm_eps": float(kw.pop("rms_norm_eps", 1e-6)),
        "tie_word_embeddings": bool(kw.pop("tie_word_embeddings", True)),
        "dropout": float(kw.pop("dropout", 0.0)),
        "eos_token_id": int(kw.pop("eos_token_id", 3)),
        "attn_backend": str(kw.pop("attn_backend", "cudnn")),
        "grad_checkpoint": bool(kw.pop("grad_checkpoint", False)),
    }
    cfg.update(kw)
    validate_cfg_1126(cfg)
    return cfg


def validate_cfg_1126(cfg):
    d = int(cfg["hidden_size"])
    h_kv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    m = int(cfg["num_q_sets"])
    if h_kv * hd != d:
        raise ValueError("无KV投影要求 num_key_value_heads(%d) × head_dim(%d) == hidden_size(%d)"
                         % (h_kv, hd, d))
    if hd % 2 != 0:
        raise ValueError("head_dim(%d) 必须是偶数" % hd)
    if cfg.get("attn_backend", "cudnn") == "cudnn" and hd % 8 != 0:
        raise ValueError("RTX 3070 cuDNN 要求 head_dim(%d) 必须是 8 的倍数" % hd)
    if int(cfg["num_attention_heads"]) != h_kv * m:
        raise ValueError("num_attention_heads(%d) != num_key_value_heads(%d) × num_q_sets(%d)"
                         % (cfg["num_attention_heads"], h_kv, m))
    if cfg["gate_mode"] not in ("head", "scalar"):
        raise ValueError("gate_mode 只支持 head / scalar")
    if int(cfg["mtp"]) not in (0, 2, 3, 4):
        raise ValueError("mtp 只支持 0/2/3/4")


def param_breakdown_1126(cfg):
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    h_kv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    m = int(cfg["num_q_sets"])
    inter = int(cfg["intermediate_size"])

    embed = int(cfg["vocab_size"]) * d * (1 if cfg.get("tie_word_embeddings", True) else 2)
    q_proj = d * (m * d)
    if cfg.get("per_head_qk_norm", True):
        qk_norm = (m * h_kv * hd) + ((h_kv * hd) if cfg.get("use_k_norm", True) else 0)
    else:
        qk_norm = hd + (hd if cfg.get("use_k_norm", True) else 0)
    gate_out = (h_kv * m) if cfg["gate_mode"] == "head" else m
    gate = d * gate_out + gate_out
    o_proj = (d * d) if cfg.get("use_o_proj", True) else 0
    layer_norms = 3 * d
    bicross = 4 * d * d
    swiglu = 3 * d * inter

    per_layer = q_proj + qk_norm + gate + o_proj + layer_norms + bicross + swiglu
    attn_total = L * (q_proj + qk_norm + gate + o_proj)
    bicross_total = L * bicross
    swiglu_total = L * swiglu
    ffn_total = bicross_total + swiglu_total
    norms_total = L * layer_norms + d
    n_mtp = max(0, int(cfg.get("mtp") or 0) - 1)
    mtp_total = n_mtp * (d + d * d)
    total = embed + attn_total + ffn_total + norms_total + mtp_total
    return {
        "embed": embed, "attn": attn_total,
        "bicross": bicross_total, "swiglu": swiglu_total, "ffn": ffn_total,
        "norms": norms_total, "mtp": mtp_total,
        "per_layer": per_layer, "total": total,
    }


def param_report_1126(model=None, cfg=None, verbose=True):
    cfg = dict(cfg or (model.cfg if model is not None else {}))
    bd = param_breakdown_1126(cfg)
    total = bd["total"]
    measured = model.num_params() if model is not None else None
    if verbose:
        print("=" * 82)
        print("1126 4Q-NoKV-DualFFN 参数账本  d=%d ｜ %d层独立 ｜ %d套Q×%d头(hd=%d)=%d Q头查%d KV头"
              % (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_q_sets"],
                 cfg["num_key_value_heads"], cfg["head_dim"], cfg["num_attention_heads"],
                 cfg["num_key_value_heads"]))
        print("  串联双FFN: BiCross(4d²) -> SwiGLU(i=%d, 6d²) ｜ 下一层Q查SwiGLU最终输出(d=%d)"
              % (cfg["intermediate_size"], cfg["hidden_size"]))
        print("  门控: %s(%d路) ｜ W_O: %s ｜ 逐头QK-Norm: %s ｜ MTP: %d ｜ native_gqa: %s"
              % (cfg["gate_mode"],
                 cfg["num_attention_heads"] if cfg["gate_mode"] == "head" else cfg["num_q_sets"],
                 "有" if cfg["use_o_proj"] else "无",
                 "有" if cfg.get("per_head_qk_norm", True) else "无",
                 int(cfg["mtp"]),
                 "是" if (cfg.get("use_native_gqa", True) and _SDPA_HAS_GQA) else "否(repeat)"))
        print("  单层 %s (%.3fM) ｜ 占比：Embed %.2fM(%.1f%%) ｜ Attn %.2fM(%.1f%%) ｜ "
              "BiCross %.2fM(%.1f%%) + SwiGLU %.2fM(%.1f%%) = FFN %.2fM(%.1f%%)"
              % ("{:,}".format(bd["per_layer"]), bd["per_layer"] / 1e6,
                 bd["embed"] / 1e6, 100.0 * bd["embed"] / total,
                 bd["attn"] / 1e6, 100.0 * bd["attn"] / total,
                 bd["bicross"] / 1e6, 100.0 * bd["bicross"] / total,
                 bd["swiglu"] / 1e6, 100.0 * bd["swiglu"] / total,
                 bd["ffn"] / 1e6, 100.0 * bd["ffn"] / total))
        print("  公式总数 %s (%.4f M) ｜ 4bit ≈ %.2f MiB ｜ bf16 ≈ %.2f MiB"
              % ("{:,}".format(total), total / 1e6, total * 0.5 / 1048576, total * 2.0 / 1048576))
        if measured is not None:
            print("  torch 实测 %s (%.4f M) ｜ 差异 %d"
                  % ("{:,}".format(measured), measured / 1e6, measured - total))
            assert measured == total, "公式 %d != 实测 %d" % (total, measured)
        print("=" * 82)
    return measured, total, bd


def kv_table_1126(cfg, contexts=(512, 1088)):
    L = int(cfg["num_hidden_layers"])
    d = int(cfg["hidden_size"])
    print("KV Cache（1126 单份干净状态，%d 层 d=%d，单位 MiB）" % (L, d))
    print("  ctx    fp16(单份)   对比常规fp16(K+V双份)")
    for c in contexts:
        single = L * c * d * 2 / 1048576
        print("  %4d     %6.2f          %6.2f" % (c, single, single * 2))


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    c = cfg_1126()
    param_report_1126(LM1126(c), c)
    kv_table_1126(c)
