# -*- coding: utf-8 -*-
"""model_arena.py —— 1125 竞技场 2 号与 3 号选手统一实现（与 1 号完全同口径、严禁改动 1 号代码）

支持两种架构（通过 cfg["arch_type"] 切换）：
  1) "4q_swiglu"（2 号选手）：
       d=256, 10 层独立, 4 套并行 Q (32 Q 头查 8 KV 头, hd=32), 无 W_K/W_V(单份干净缓存),
       32 路逐头动态门控, 共享 W_O, 去掉 BiCross 改为等参纯 SwiGLU (i=856, 3.34×) -> 12.08M
  2) "std_gqa"（3 号选手，传统现代 GQA Transformer 基线）：
       d=256, 10 层独立, 8 Q 头 / 2 KV 头 (hd=32), 独立 W_Q/W_K/W_V/W_O + QK-Norm + RoPE,
       等参纯 SwiGLU (i=1080, 4.22×) -> 12.06M

均采用 PyTorch 原生融合 F.rms_norm + 半维原生 dtype 高速 RoPE + 原生 GQA cuDNN SDPA，
确保与 1 号选手在完全同一套底层优化下公平竞技。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as _ckpt

from model_1126 import (
    IGNORE,
    _SDPA_HAS_GQA,
    sdpa_ctx,
    RMSNorm,
    precompute_freqs_cis,
    apply_rope,
    repeat_kv_heads,
    ParallelQNoKVAttention,
    SwiGLUFFN,
    _MtpHead,
)


# ======================================================================
# 3 号选手注意力：标准现代 GQA（独立 W_Q / W_K / W_V / W_O + QK-Norm + RoPE）
# ======================================================================
class StandardGQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d = int(cfg["hidden_size"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv_heads = int(cfg["num_key_value_heads"])
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = int(cfg["head_dim"])
        self.use_native_gqa = bool(cfg.get("use_native_gqa", True)) and _SDPA_HAS_GQA
        self.attn_backend = str(cfg.get("attn_backend", "cudnn"))
        self.dropout = float(cfg.get("dropout", 0.0))

        self.q_proj = nn.Linear(self.d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.resid_dropout = nn.Dropout(self.dropout)

    @staticmethod
    def _causal_mask(q_len, kv_len, device, dtype):
        off = kv_len - q_len
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=off)
        m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        return m.masked_fill(~causal, torch.finfo(dtype).min)

    def forward(self, x_norm, q_rope, k_rope=None, past_kv=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x_norm.shape
        xq = self.q_norm(self.q_proj(x_norm).view(bsz, seq_len, self.n_heads, self.head_dim))
        xk = self.k_norm(self.k_proj(x_norm).view(bsz, seq_len, self.n_kv_heads, self.head_dim))
        xv = self.v_proj(x_norm).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rope(xq, q_rope[0], q_rope[1])
        xk = apply_rope(xk, q_rope[0], q_rope[1])

        if past_kv is not None:
            xk = torch.cat((past_kv[0], xk), dim=1)
            xv = torch.cat((past_kv[1], xv), dim=1)
        present = (xk, xv) if use_cache else None
        kv_len = xk.shape[1]

        xq = xq.transpose(1, 2)
        xk_t = xk.transpose(1, 2)
        xv_t = xv.transpose(1, 2)

        p = self.dropout if self.training else 0.0
        single_decode_no_mask = (seq_len == 1) and (attention_mask is None)
        needs_mask = (not single_decode_no_mask) and ((kv_len != seq_len) or (attention_mask is not None))
        with sdpa_ctx(self.attn_backend, xq.dtype):
            if self.use_native_gqa:
                if single_decode_no_mask:
                    out = F.scaled_dot_product_attention(
                        xq, xk_t, xv_t, dropout_p=p, is_causal=False, enable_gqa=True)
                elif not needs_mask:
                    out = F.scaled_dot_product_attention(
                        xq, xk_t, xv_t, dropout_p=p, is_causal=True, enable_gqa=True)
                else:
                    m = self._causal_mask(seq_len, kv_len, xq.device, xq.dtype)
                    if attention_mask is not None:
                        m = m.masked_fill(attention_mask[:, None, None, :].to(xq.device) == 0,
                                          torch.finfo(xq.dtype).min)
                    out = F.scaled_dot_product_attention(
                        xq, xk_t, xv_t, attn_mask=m, dropout_p=p, enable_gqa=True)
            else:
                xk_r = repeat_kv_heads(xk_t, self.n_rep)
                xv_r = repeat_kv_heads(xv_t, self.n_rep)
                if single_decode_no_mask:
                    out = F.scaled_dot_product_attention(xq, xk_r, xv_r, dropout_p=p, is_causal=False)
                elif not needs_mask:
                    out = F.scaled_dot_product_attention(xq, xk_r, xv_r, dropout_p=p, is_causal=True)
                else:
                    m = self._causal_mask(seq_len, kv_len, xq.device, xq.dtype)
                    if attention_mask is not None:
                        m = m.masked_fill(attention_mask[:, None, None, :].to(xq.device) == 0,
                                          torch.finfo(xq.dtype).min)
                    out = F.scaled_dot_product_attention(xq, xk_r, xv_r, attn_mask=m, dropout_p=p)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.d)
        return self.resid_dropout(self.o_proj(out)), present


# ======================================================================
# 2 号 / 3 号单层 Block（Pre-Norm -> Attn -> 残差 -> Pre-Norm -> SwiGLU -> 残差）
# ======================================================================
class ArenaBlock(nn.Module):
    def __init__(self, cfg, idx=0):
        super().__init__()
        d = int(cfg["hidden_size"])
        eps = float(cfg["rms_norm_eps"])
        self.idx = idx
        self.arch_type = cfg["arch_type"]
        self.input_layernorm = RMSNorm(d, eps=eps)
        if self.arch_type == "4q_swiglu":
            self.self_attn = ParallelQNoKVAttention(cfg)
        else:
            self.self_attn = StandardGQAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.swiglu = SwiGLUFFN(cfg)

    def forward(self, hidden_states, q_rope, k_rope, past_kv=None, use_cache=False, attention_mask=None):
        x_norm = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            x_norm, q_rope, k_rope, past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.swiglu(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class ArenaBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.arch_type = cfg["arch_type"]
        self.n_layers = int(cfg["num_hidden_layers"])
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        d = int(cfg["hidden_size"])

        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), d)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.layers = nn.ModuleList([ArenaBlock(cfg, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(d, eps=float(cfg["rms_norm_eps"]))

        cos, sin = precompute_freqs_cis(
            int(cfg["head_dim"]), int(cfg["max_position_embeddings"]),
            rope_base=float(cfg["rope_theta"]), rope_scaling=cfg.get("rope_scaling"))
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    @staticmethod
    def _past_len(past0):
        if past0 is None:
            return 0
        if isinstance(past0, (tuple, list)):
            return past0[0].shape[1]
        return past0.shape[1]

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
            past_len = self._past_len(past[0])

        if position_offset is None:
            position_offset = past_len
        kv_len = past_len + seq_len
        k_start = position_offset - past_len
        if k_start < 0 or position_offset + seq_len > self.freqs_cos.shape[0] or k_start + kv_len > self.freqs_cos.shape[0]:
            raise ValueError("序列位置越界: k_start=%d, q_end=%d, max=%d"
                             % (k_start, position_offset + seq_len, self.freqs_cos.shape[0]))

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


class LMArena(nn.Module):
    """2 号（4q_swiglu）与 3 号（std_gqa）统一模型封装。"""

    def __init__(self, cfg):
        super().__init__()
        cfg = cfg_arena(**cfg)
        self.cfg = cfg
        self.model = ArenaBackbone(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        n_mtp = max(0, int(cfg.get("mtp") or 0) - 1)
        self.mtp_heads = nn.ModuleList([_MtpHead(cfg) for _ in range(n_mtp)])

        self.apply(self._init_weights)
        if cfg.get("resid_init_scale", True):
            r_std = 0.02 / math.sqrt(2.0 * cfg["num_hidden_layers"])
            ffn_std = r_std / math.sqrt(float(cfg["intermediate_size"]) / float(cfg["hidden_size"]))
            for layer in self.model.layers:
                if getattr(layer.self_attn, "o_proj", None) is not None:
                    nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=r_std)
                nn.init.normal_(layer.swiglu.down_proj.weight, mean=0.0, std=ffn_std)
        if cfg["arch_type"] == "4q_swiglu":
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
        if self.cfg["arch_type"] == "4q_swiglu":
            for layer in self.model.layers:
                layer.self_attn.enable_gate_stats()

    def pop_gate_stats(self):
        if self.cfg["arch_type"] == "4q_swiglu":
            return [layer.self_attn.pop_gate_stats() for layer in self.model.layers]
        return None

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
        stop_ids = set(int(x) for x in eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {int(eos_token_id)}
        ids = input_ids
        if not use_cache:
            past_key_values = None
        use_attn_mask = attention_mask is not None
        max_pos = int(self.cfg["max_position_embeddings"])
        generated = []
        for _ in range(max_new_tokens):
            if ids.shape[1] >= max_pos:
                break
            past_len = ArenaBackbone._past_len(past_key_values[0]) if (use_cache and past_key_values) else 0
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
# 配置 / 校验 / 参数账本
# ======================================================================
def cfg_arena(**kw):
    arch = str(kw.pop("arch_type", "4q_swiglu")).lower()
    d = int(kw.pop("hidden_size", 256))
    if arch == "4q_swiglu":
        kv_heads = int(kw.pop("num_key_value_heads", 8))
        q_sets = int(kw.pop("num_q_sets", 4))
        attn_heads = int(kw.pop("num_attention_heads", kv_heads * q_sets))
        hd = int(kw.pop("head_dim", d // kv_heads))
        inter = int(kw.pop("intermediate_size", 856))
    else:
        attn_heads = int(kw.pop("num_attention_heads", 8))
        kv_heads = int(kw.pop("num_key_value_heads", 2))
        q_sets = 1
        hd = int(kw.pop("head_dim", d // attn_heads))
        inter = int(kw.pop("intermediate_size", 1080))
    cfg = {
        "arch_type": arch,
        "vocab_size": int(kw.pop("vocab_size", 8302)),
        "hidden_size": d,
        "num_hidden_layers": int(kw.pop("num_hidden_layers", 10)),
        "num_attention_heads": attn_heads,
        "num_key_value_heads": kv_heads,
        "num_q_sets": q_sets,
        "head_dim": hd,
        "gate_mode": str(kw.pop("gate_mode", "head")).lower(),
        "use_o_proj": bool(kw.pop("use_o_proj", True)),
        "use_k_norm": bool(kw.pop("use_k_norm", True)),
        "per_head_qk_norm": bool(kw.pop("per_head_qk_norm", True)),
        "use_native_gqa": bool(kw.pop("use_native_gqa", True)),
        "resid_init_scale": bool(kw.pop("resid_init_scale", True)),
        "intermediate_size": inter,
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
    validate_cfg_arena(cfg)
    return cfg


def validate_cfg_arena(cfg):
    arch = cfg["arch_type"]
    if arch not in ("4q_swiglu", "std_gqa"):
        raise ValueError("arch_type 只支持 4q_swiglu / std_gqa（当前 %s）" % arch)
    d = int(cfg["hidden_size"])
    hd = int(cfg["head_dim"])
    if hd % 8 != 0 and cfg.get("attn_backend", "cudnn") == "cudnn":
        raise ValueError("cuDNN 要求 head_dim(%d) 必须是 8 的倍数" % hd)
    if arch == "4q_swiglu":
        h_kv, m = int(cfg["num_key_value_heads"]), int(cfg["num_q_sets"])
        if h_kv * hd != d or int(cfg["num_attention_heads"]) != h_kv * m:
            raise ValueError("4q_swiglu 维度不匹配")
    else:
        h_q, h_kv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        if h_q * hd != d or h_q % h_kv != 0:
            raise ValueError("std_gqa 维度不匹配")


def param_breakdown_arena(cfg):
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    hd = int(cfg["head_dim"])
    inter = int(cfg["intermediate_size"])
    embed = int(cfg["vocab_size"]) * d * (1 if cfg.get("tie_word_embeddings", True) else 2)
    if cfg["arch_type"] == "4q_swiglu":
        h_kv, m = int(cfg["num_key_value_heads"]), int(cfg["num_q_sets"])
        q_proj = d * (m * d)
        qk_norm = ((m * h_kv * hd) + ((h_kv * hd) if cfg.get("use_k_norm", True) else 0)) \
            if cfg.get("per_head_qk_norm", True) else (hd + (hd if cfg.get("use_k_norm", True) else 0))
        gate_out = (h_kv * m) if cfg["gate_mode"] == "head" else m
        gate = d * gate_out + gate_out
        o_proj = (d * d) if cfg.get("use_o_proj", True) else 0
        attn_per = q_proj + qk_norm + gate + o_proj
    else:
        h_q, h_kv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        attn_per = d * (h_q * hd) + 2 * d * (h_kv * hd) + (h_q * hd) * d + 2 * hd
    layer_norms = 2 * d
    ffn_per = 3 * d * inter
    per_layer = attn_per + layer_norms + ffn_per
    attn_total = L * attn_per
    ffn_total = L * ffn_per
    norms_total = L * layer_norms + d
    n_mtp = max(0, int(cfg.get("mtp") or 0) - 1)
    mtp_total = n_mtp * (d + d * d)
    total = embed + attn_total + ffn_total + norms_total + mtp_total
    return {
        "embed": embed, "attn": attn_total, "ffn": ffn_total,
        "norms": norms_total, "mtp": mtp_total,
        "per_layer": per_layer, "total": total,
    }


def param_report_arena(model=None, cfg=None, verbose=True):
    cfg = dict(cfg or (model.cfg if model is not None else {}))
    bd = param_breakdown_arena(cfg)
    total = bd["total"]
    measured = model.num_params() if model is not None else None
    if verbose:
        print("=" * 80)
        tag = "2号选手 4Q-NoKV-SwiGLU" if cfg["arch_type"] == "4q_swiglu" else "3号选手 传统GQA-SwiGLU"
        print("%s  d=%d ｜ %d层独立 ｜ Q头=%d KV头=%d (hd=%d) ｜ SwiGLU i=%d (%.2f×)"
              % (tag, cfg["hidden_size"], cfg["num_hidden_layers"],
                 cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"],
                 cfg["intermediate_size"], cfg["intermediate_size"] / float(cfg["hidden_size"])))
        print("  单层 %s (%.3fM) ｜ 占比：Embed %.2fM(%.1f%%) ｜ Attn %.2fM(%.1f%%) ｜ FFN %.2fM(%.1f%%)"
              % ("{:,}".format(bd["per_layer"]), bd["per_layer"] / 1e6,
                 bd["embed"] / 1e6, 100.0 * bd["embed"] / total,
                 bd["attn"] / 1e6, 100.0 * bd["attn"] / total,
                 bd["ffn"] / 1e6, 100.0 * bd["ffn"] / total))
        print("  公式总数 %s (%.4f M) ｜ 4bit ≈ %.2f MiB ｜ bf16 ≈ %.2f MiB"
              % ("{:,}".format(total), total / 1e6, total * 0.5 / 1048576, total * 2.0 / 1048576))
        if measured is not None:
            print("  torch 实测 %s (%.4f M) ｜ 差异 %d"
                  % ("{:,}".format(measured), measured / 1e6, measured - total))
            assert measured == total, "公式 %d != 实测 %d" % (total, measured)
        print("=" * 80)
    return measured, total, bd
