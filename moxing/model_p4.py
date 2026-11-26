# -*- coding: utf-8 -*-
"""model_p4.py —— 1125 竞技场 4 号选手（接法 B：传统 2-KV GQA + 真·双线性门控 SwiGLU）

核心结构（d=256, 10 层独立, 总参 12.04M，与 1/2/3 号严格同参）：
  · 注意力：与 3 号完全一致（传统现代 GQA：8 Q 头 / 2 KV 头, hd=32, 独立 W_Q/W_K/W_V/W_O + QK-Norm + RoPE）
  · FFN（BiCrossGateFFN，接法 B）：
      1) 用 BiCross 双分支做二阶跨维交叉：
           q1, q2 = self.q(x).chunk(2, dim=-1)             # 2d²
           b_feat = cat(GELU(q1 * x), SiLU(q2 * x), -1)    # 2d 维交叉特征
      2) 由 2d 交叉特征生成宽维高级开关：
           gate = self.gate_proj(b_feat)                   # 2d -> i (=680)
      3) 开关 × 升维仓库 -> 降维输出：
           y = self.down_proj(gate * self.up_proj(x))      # i -> d
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as _ckpt

from model_1126 import (
    IGNORE,
    RMSNorm,
    precompute_freqs_cis,
    _MtpHead,
)
from model_arena import StandardGQAAttention, ArenaBackbone


class BiCrossGateFFN(nn.Module):
    """接法 B：双线性交叉高级开关 × 升维仓库 -> 降维输出（2d² + 4d·i）。"""

    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["hidden_size"])
        i = int(cfg["intermediate_size"])
        self.act1 = str(cfg.get("qbilinear_act1") or "gelu").lower()
        self.act2 = str(cfg.get("qbilinear_act2") or "silu").lower()
        self.q = nn.Linear(d, 2 * d, bias=False)
        self.gate_proj = nn.Linear(2 * d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

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
        gate = self.gate_proj(torch.cat((a, c), dim=-1))
        return self.down_proj(gate * self.up_proj(x))


class P4Block(nn.Module):
    def __init__(self, cfg, idx=0, shared_ffn=None):
        super().__init__()
        d = int(cfg["hidden_size"])
        eps = float(cfg["rms_norm_eps"])
        self.idx = idx
        self.input_layernorm = RMSNorm(d, eps=eps)
        self.self_attn = StandardGQAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.ffn = None if shared_ffn is not None else BiCrossGateFFN(cfg)

    def forward(self, hidden_states, q_rope, k_rope, past_kv=None, use_cache=False,
                attention_mask=None, ffn_override=None):
        x_norm = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            x_norm, q_rope, k_rope, past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        mlp = ffn_override if ffn_override is not None else self.ffn
        hidden_states = hidden_states + mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class P4Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers = int(cfg["num_hidden_layers"])
        self.ffn_share = int(cfg.get("ffn_share_group") or 1)
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        d = int(cfg["hidden_size"])
        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), d)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        if self.ffn_share > 1:
            n_groups = (self.n_layers + self.ffn_share - 1) // self.ffn_share
            self.ffn_groups = nn.ModuleList([BiCrossGateFFN(cfg) for _ in range(n_groups)])
            self.layers = nn.ModuleList([P4Block(cfg, i, shared_ffn=True) for i in range(self.n_layers)])
        else:
            self.ffn_groups = None
            self.layers = nn.ModuleList([P4Block(cfg, i) for i in range(self.n_layers)])
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
            for i, layer in enumerate(self.layers):
                ffn_ov = self.ffn_groups[i // self.ffn_share] if self.ffn_groups is not None else None
                def _fn(h, _lay=layer, _fo=ffn_ov):
                    out_h, _ = _lay(h, q_rope, k_rope, None, False, attention_mask, ffn_override=_fo)
                    return out_h
                hidden = _ckpt(_fn, hidden, use_reentrant=False)
            return self.norm(hidden), []

        presents = []
        for i, (layer, pkv) in enumerate(zip(self.layers, past)):
            ffn_ov = self.ffn_groups[i // self.ffn_share] if self.ffn_groups is not None else None
            hidden, pres = layer(hidden, q_rope, k_rope, past_kv=pkv,
                                 use_cache=use_cache, attention_mask=attention_mask,
                                 ffn_override=ffn_ov)
            if use_cache:
                presents.append(pres)
        return self.norm(hidden), presents


class LMP4(nn.Module):
    """4 号选手语言模型（传统 2-KV GQA + 双线性交叉门控 SwiGLU）。"""

    def __init__(self, cfg):
        super().__init__()
        cfg = cfg_p4(**cfg)
        self.cfg = cfg
        self.model = P4Backbone(cfg)
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
                nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=r_std)
                if layer.ffn is not None:
                    nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=ffn_std)
            if self.model.ffn_groups is not None:
                for grp in self.model.ffn_groups:
                    nn.init.normal_(grp.down_proj.weight, mean=0.0, std=ffn_std)

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


def cfg_p4(**kw):
    d = int(kw.pop("hidden_size", 256))
    attn_heads = int(kw.pop("num_attention_heads", 8))
    cfg = {
        "arch_type": "std_gqa_bicross_gate",
        "vocab_size": int(kw.pop("vocab_size", 8302)),
        "hidden_size": d,
        "num_hidden_layers": int(kw.pop("num_hidden_layers", 10)),
        "num_attention_heads": attn_heads,
        "num_key_value_heads": int(kw.pop("num_key_value_heads", 2)),
        "head_dim": int(kw.pop("head_dim", d // attn_heads)),
        "intermediate_size": int(kw.pop("intermediate_size", 680)),
        "ffn_share_group": int(kw.pop("ffn_share_group", 1) or 1),
        "qbilinear_act1": str(kw.pop("qbilinear_act1", "gelu")).lower(),
        "qbilinear_act2": str(kw.pop("qbilinear_act2", "silu")).lower(),
        "use_native_gqa": bool(kw.pop("use_native_gqa", True)),
        "resid_init_scale": bool(kw.pop("resid_init_scale", True)),
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
    return cfg


def param_report_p4(model=None, cfg=None, verbose=True):
    cfg = dict(cfg or (model.cfg if model is not None else {}))
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    share = int(cfg.get("ffn_share_group") or 1)
    n_ffn = (L + share - 1) // share if share > 1 else L
    h_q, h_kv, hd = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
    inter = int(cfg["intermediate_size"])
    embed = int(cfg["vocab_size"]) * d * (1 if cfg.get("tie_word_embeddings", True) else 2)
    attn_per = d * (h_q * hd) + 2 * d * (h_kv * hd) + (h_q * hd) * d + 2 * hd
    ffn_per = 2 * d * d + 4 * d * inter
    norms_per = 2 * d
    total = embed + L * (attn_per + norms_per) + n_ffn * ffn_per + d
    measured = model.num_params() if model is not None else None
    if verbose:
        print("=" * 82)
        tag = ("5号选手 传统2KV-GQA + 双线性交叉门控(每%d层共享, %d组)" % (share, n_ffn)) if share > 1 \
            else "4号选手 传统2KV-GQA + 双线性交叉门控SwiGLU(接法B)"
        print("%s  d=%d ｜ %d层 ｜ i=%d (%.2f×)" % (tag, d, L, inter, inter / float(d)))
        print("  占比：Embed %.2fM(%.1f%%) ｜ Attn %.2fM(%.1f%%) ｜ FFN(%d组) %.2fM(%.1f%%)"
              % (embed / 1e6, 100.0 * embed / total,
                 (L * attn_per) / 1e6, 100.0 * (L * attn_per) / total,
                 n_ffn, (n_ffn * ffn_per) / 1e6, 100.0 * (n_ffn * ffn_per) / total))
        print("  公式总数 %s (%.4f M) ｜ 4bit ≈ %.2f MiB ｜ bf16 ≈ %.2f MiB"
              % ("{:,}".format(total), total / 1e6, total * 0.5 / 1048576, total * 2.0 / 1048576))
        if measured is not None:
            print("  torch 实测 %s (%.4f M) ｜ 差异 %d"
                  % ("{:,}".format(measured), measured / 1e6, measured - total))
            assert measured == total, "公式 %d != 实测 %d" % (total, measured)
        print("=" * 82)
    return measured, total
