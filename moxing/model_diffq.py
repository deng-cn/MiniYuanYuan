# -*- coding: utf-8 -*-
"""model_diffq.py —— 1127 挑战赛 2 号选手：相邻双 Q 共享池 + 差分去噪注意力 (Diff-Attn) + 标准 SwiGLU

核心结构（严格对标 1 号传统基线 p3：d=256, 10 层, 8Q/2KV GQA, hd=32, SwiGLU）：
  1. 相邻双 Q 共享池（Adjacent Sliding Q-Pool）：
     - 10 层共享 11 个标准 W_Q 投影矩阵 {Q_0, Q_1, ..., Q_10}；
     - 第 l 层取紧邻的一对 (Q_l, Q_{l+1})，严禁跨远层或首尾回环共享，消除浅/深层梯度打架；
     - 每层保留各自独立的逐头 MultiHeadRMSNorm（q1_norm / q2_norm），让复用的 Q 矩阵在不同层拥有独立温度与通道缩放。
  2. 单次融合 GQA 差分去噪（Differential Attention, Ye et al. ICLR 2025）：
     - 将 (q1, q2) 按 KV 组交错拼接为 16 个 Q 头，单次 cuDNN GQA SDPA 同步查同一份 2-KV Cache（推理 KV Cache 零增加）；
     - 差分混合：diff = y1 - lambda_l * y2，其中 lambda_l 由层内 4 个 1D 向量重参数化，按深度递增初始化
       lambda_init(l) = 0.8 - 0.6 * exp(-0.3 * l)；
     - 经逐头 GroupRMSNorm（sub_norm）并乘 (1 - lambda_init) 恢复标准残差尺度，再过标准 W_O。
  3. FFN 及后续结构与 1 号传统基线完全一致（仅将 SwiGLU intermediate_size 从 1080 微调至 1072 对齐 12.07M 等参）。
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
    MultiHeadRMSNorm,
    precompute_freqs_cis,
    apply_rope,
    repeat_kv_heads,
    SwiGLUFFN,
    _MtpHead,
)


def lambda_init_fn(layer_idx):
    return 0.8 - 0.6 * math.exp(-0.3 * float(layer_idx))


class DiffQPoolAttention(nn.Module):
    """相邻共享双 Q 池 + 差分去噪 GQA 注意力（复用共享池中 (Q_l, Q_{l+1})，KV Cache 零膨胀）。"""

    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.d = int(cfg["hidden_size"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv_heads = int(cfg["num_key_value_heads"])
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = int(cfg["head_dim"])
        self.use_native_gqa = bool(cfg.get("use_native_gqa", True)) and _SDPA_HAS_GQA
        self.attn_backend = str(cfg.get("attn_backend", "cudnn"))
        self.dropout = float(cfg.get("dropout", 0.0))
        eps = float(cfg["rms_norm_eps"])

        # K / V / O 独立每层一份（与 1 号基线完全一致）
        self.k_proj = nn.Linear(self.d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

        # 每层独立的逐头 q1/q2 归一化（避免共享 W_Q 在相邻两层尺度/温度打架）
        self.q1_norm = MultiHeadRMSNorm((1, 1, self.n_heads, self.head_dim), eps=eps)
        self.q2_norm = MultiHeadRMSNorm((1, 1, self.n_heads, self.head_dim), eps=eps)

        # Diff-Attn 差分去噪核心：重参数化 lambda_l（全 1D 向量，自动免 weight decay）+ 逐头 GroupRMSNorm
        self.lambda_init = float(lambda_init_fn(self.layer_idx))
        self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim).normal_(mean=0.0, std=0.1))
        self.sub_norm = MultiHeadRMSNorm((1, 1, self.n_heads, self.head_dim), eps=eps)

        self.resid_dropout = nn.Dropout(self.dropout)

    def current_lambda(self):
        l1 = torch.exp(torch.sum(self.lambda_q1.float() * self.lambda_k1.float()))
        l2 = torch.exp(torch.sum(self.lambda_q2.float() * self.lambda_k2.float()))
        return l1 - l2 + self.lambda_init

    @staticmethod
    def _causal_mask(q_len, kv_len, device, dtype):
        off = kv_len - q_len
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=off)
        m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        return m.masked_fill(~causal, torch.finfo(dtype).min)

    def forward(self, x_norm, q_proj_a, q_proj_b, q_rope, past_kv=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x_norm.shape

        # 从共享池读取相邻两路 Q，各自过本层独立逐头 Norm 与 RoPE
        q1 = self.q1_norm(q_proj_a(x_norm).view(bsz, seq_len, self.n_heads, self.head_dim))
        q2 = self.q2_norm(q_proj_b(x_norm).view(bsz, seq_len, self.n_heads, self.head_dim))
        xk = self.k_norm(self.k_proj(x_norm).view(bsz, seq_len, self.n_kv_heads, self.head_dim))
        xv = self.v_proj(x_norm).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q1 = apply_rope(q1, q_rope[0], q_rope[1])
        q2 = apply_rope(q2, q_rope[0], q_rope[1])
        xk = apply_rope(xk, q_rope[0], q_rope[1])

        if past_kv is not None:
            xk = torch.cat((past_kv[0], xk), dim=1)
            xv = torch.cat((past_kv[1], xv), dim=1)
        present = (xk, xv) if use_cache else None
        kv_len = xk.shape[1]

        # 按 KV 组交错拼接 [B, H_kv, 2, n_rep, T, D] -> [B, 2*H, T, D]，单次 SDPA 同时算完两路
        q1_g = q1.transpose(1, 2).view(bsz, self.n_kv_heads, self.n_rep, seq_len, self.head_dim)
        q2_g = q2.transpose(1, 2).view(bsz, self.n_kv_heads, self.n_rep, seq_len, self.head_dim)
        q_all = torch.stack((q1_g, q2_g), dim=2).reshape(bsz, 2 * self.n_heads, seq_len, self.head_dim)

        xk_t = xk.transpose(1, 2)
        xv_t = xv.transpose(1, 2)

        p = self.dropout if self.training else 0.0
        single_decode_no_mask = (seq_len == 1) and (attention_mask is None)
        needs_mask = (not single_decode_no_mask) and ((kv_len != seq_len) or (attention_mask is not None))

        with sdpa_ctx(self.attn_backend, q_all.dtype):
            if self.use_native_gqa:
                if single_decode_no_mask:
                    out_all = F.scaled_dot_product_attention(
                        q_all, xk_t, xv_t, dropout_p=p, is_causal=False, enable_gqa=True)
                elif not needs_mask:
                    out_all = F.scaled_dot_product_attention(
                        q_all, xk_t, xv_t, dropout_p=p, is_causal=True, enable_gqa=True)
                else:
                    m = self._causal_mask(seq_len, kv_len, q_all.device, q_all.dtype)
                    if attention_mask is not None:
                        m = m.masked_fill(attention_mask[:, None, None, :].to(q_all.device) == 0,
                                          torch.finfo(q_all.dtype).min)
                    out_all = F.scaled_dot_product_attention(
                        q_all, xk_t, xv_t, attn_mask=m, dropout_p=p, enable_gqa=True)
            else:
                xk_r = repeat_kv_heads(xk_t, 2 * self.n_rep)
                xv_r = repeat_kv_heads(xv_t, 2 * self.n_rep)
                if single_decode_no_mask:
                    out_all = F.scaled_dot_product_attention(q_all, xk_r, xv_r, dropout_p=p, is_causal=False)
                elif not needs_mask:
                    out_all = F.scaled_dot_product_attention(q_all, xk_r, xv_r, dropout_p=p, is_causal=True)
                else:
                    m = self._causal_mask(seq_len, kv_len, q_all.device, q_all.dtype)
                    if attention_mask is not None:
                        m = m.masked_fill(attention_mask[:, None, None, :].to(q_all.device) == 0,
                                          torch.finfo(q_all.dtype).min)
                    out_all = F.scaled_dot_product_attention(q_all, xk_r, xv_r, attn_mask=m, dropout_p=p)

        out_g = out_all.view(bsz, self.n_kv_heads, 2, self.n_rep, seq_len, self.head_dim)
        y1 = out_g[:, :, 0].reshape(bsz, self.n_heads, seq_len, self.head_dim)
        y2 = out_g[:, :, 1].reshape(bsz, self.n_heads, seq_len, self.head_dim)

        lam = self.current_lambda().to(y1.dtype)
        diff = (y1 - lam * y2).transpose(1, 2)                      # [B, T, H, D]
        diff = self.sub_norm(diff) * (1.0 - self.lambda_init)       # GroupRMSNorm + 残差标定
        out = diff.reshape(bsz, seq_len, self.d)
        return self.resid_dropout(self.o_proj(out)), present


class DiffQPoolBlock(nn.Module):
    """单层 Block：Pre-Norm -> DiffQPoolAttention -> 残差 -> Pre-Norm -> SwiGLU -> 残差。"""

    def __init__(self, cfg, idx=0):
        super().__init__()
        d = int(cfg["hidden_size"])
        eps = float(cfg["rms_norm_eps"])
        self.idx = idx
        self.input_layernorm = RMSNorm(d, eps=eps)
        self.self_attn = DiffQPoolAttention(cfg, layer_idx=idx)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.swiglu = SwiGLUFFN(cfg)

    def forward(self, hidden_states, q_proj_a, q_proj_b, q_rope, past_kv=None, use_cache=False, attention_mask=None):
        x_norm = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            x_norm, q_proj_a, q_proj_b, q_rope,
            past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.swiglu(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class DiffQPoolBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers = int(cfg["num_hidden_layers"])
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        d = int(cfg["hidden_size"])
        q_out = int(cfg["num_attention_heads"]) * int(cfg["head_dim"])
        self.q_pool_size = int(cfg.get("q_pool_size") or (self.n_layers + 1))
        assert self.q_pool_size >= self.n_layers + 1, "相邻无回环共享至少需要 L+1 个 Q 矩阵"

        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), d)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.q_pool = nn.ModuleList([nn.Linear(d, q_out, bias=False) for _ in range(self.q_pool_size)])
        self.layers = nn.ModuleList([DiffQPoolBlock(cfg, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(d, eps=float(cfg["rms_norm_eps"]))

        cos, sin = precompute_freqs_cis(
            int(cfg["head_dim"]), int(cfg["max_position_embeddings"]),
            rope_base=float(cfg["rope_theta"]), rope_scaling=cfg.get("rope_scaling"))
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, position_offset=None):
        _, seq_len = input_ids.shape
        hidden = self.dropout(self.embed_tokens(input_ids))
        past = past_key_values if past_key_values is not None else [None] * self.n_layers

        if position_offset is not None:
            q_start = int(position_offset)
        elif past[0] is not None:
            q_start = int(past[0][0].shape[1])
        else:
            q_start = 0

        q_rope = (self.freqs_cos[q_start:q_start + seq_len],
                  self.freqs_sin[q_start:q_start + seq_len])

        presents = [] if use_cache else None
        for i, (layer, pkv) in enumerate(zip(self.layers, past)):
            qa = self.q_pool[i]
            qb = self.q_pool[i + 1]
            if self.grad_checkpoint and self.training and not use_cache:
                hidden, _ = _ckpt(
                    lambda h, a=qa, b=qb, ly=layer: ly(
                        h, a, b, q_rope, past_kv=None, use_cache=False, attention_mask=attention_mask),
                    hidden, use_reentrant=False)
            else:
                hidden, pres = layer(
                    hidden, qa, qb, q_rope,
                    past_kv=pkv, use_cache=use_cache, attention_mask=attention_mask)
                if use_cache:
                    presents.append(pres)
        return self.norm(hidden), presents


class LMDiffQPool(nn.Module):
    """1127 挑战赛 2 号选手：相邻双 Q 共享池 + 差分去噪 GQA + SwiGLU。"""

    def __init__(self, cfg):
        super().__init__()
        cfg = cfg_diffq(**cfg)
        self.cfg = cfg
        self.model = DiffQPoolBackbone(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        n_mtp = max(0, int(cfg.get("mtp") or 0) - 1)
        self.mtp_heads = nn.ModuleList([_MtpHead(cfg) for _ in range(n_mtp)])

        self.apply(self._init_weights)
        # 恢复 lambda_q1/k1/q2/k2 的 std=0.1（避免被 _init_weights 覆盖）
        for layer in self.model.layers:
            attn = layer.self_attn
            nn.init.normal_(attn.lambda_q1, mean=0.0, std=0.1)
            nn.init.normal_(attn.lambda_k1, mean=0.0, std=0.1)
            nn.init.normal_(attn.lambda_q2, mean=0.0, std=0.1)
            nn.init.normal_(attn.lambda_k2, mean=0.0, std=0.1)

        if cfg.get("resid_init_scale", True):
            r_std = 0.02 / math.sqrt(2.0 * cfg["num_hidden_layers"])
            ffn_std = r_std / math.sqrt(float(cfg["intermediate_size"]) / float(cfg["hidden_size"]))
            for layer in self.model.layers:
                nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=r_std)
                nn.init.normal_(layer.swiglu.down_proj.weight, mean=0.0, std=ffn_std)

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

    def diff_lambda_summary(self):
        vals = [float(ly.self_attn.current_lambda().detach().cpu()) for ly in self.model.layers]
        return vals

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
                ce = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=IGNORE)
            else:
                ce = logits.sum() * 0.0
            loss = ce
            if len(self.mtp_heads) > 0:
                ws = list(self.cfg.get("mtp_weights") or [0.2] * len(self.mtp_heads))
                h = hidden
                tot = 0.0
                for idx, head in enumerate(self.mtp_heads):
                    shift = idx + 2
                    if input_ids.shape[1] <= shift:
                        break
                    tok_emb = self.model.embed_tokens(input_ids[:, shift - 1:-1])
                    h = head(h[:, :-1, :], tok_emb)
                    yl = labels[:, shift:].contiguous()
                    if (yl != IGNORE).any():
                        ml = F.cross_entropy(self.lm_head(h).reshape(-1, logits.size(-1)),
                                             yl.reshape(-1), ignore_index=IGNORE)
                        tot = tot + (ws[idx] if idx < len(ws) else 0.1) * ml
                mtp_loss = tot
                loss = ce + tot
        return {"logits": logits, "loss": loss, "mtp_loss": mtp_loss, "past_key_values": presents}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.7, top_p=0.9, top_k=50,
                 eos_token_id=None, repetition_penalty=1.1, no_repeat_ngram_size=0,
                 do_sample=True, use_cache=True, attention_mask=None, context_len=0):
        stop_ids = set()
        if eos_token_id is None:
            eos_token_id = self.cfg.get("eos_token_id")
        if isinstance(eos_token_id, (list, tuple, set)):
            stop_ids.update(int(x) for x in eos_token_id)
        elif eos_token_id is not None:
            stop_ids.add(int(eos_token_id))
        ids = input_ids
        past_key_values = None
        generated = []
        max_pos = int(self.cfg["max_position_embeddings"])
        use_attn_mask = attention_mask is not None
        for _ in range(max_new_tokens):
            if ids.shape[1] >= max_pos:
                break
            step_in = ids[:, -1:] if (use_cache and past_key_values is not None) else ids
            out = self(step_in, attention_mask=attention_mask if use_attn_mask else None,
                       past_key_values=past_key_values, use_cache=use_cache, logits_to_keep=1)
            logits = out["logits"][:, -1, :].float()
            if repetition_penalty and repetition_penalty != 1.0:
                hist = ids[0, context_len:] if context_len < ids.shape[1] else ids[0]
                for tid in set(hist.tolist()):
                    logits[0, tid] = (logits[0, tid] / repetition_penalty if logits[0, tid] > 0
                                      else logits[0, tid] * repetition_penalty)
            if no_repeat_ngram_size and no_repeat_ngram_size >= 2 and ids.shape[1] >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                seq = ids[0].tolist()
                prefix = tuple(seq[-(n - 1):])
                for i in range(len(seq) - n + 1):
                    if tuple(seq[i:i + n - 1]) == prefix:
                        logits[0, seq[i + n - 1]] = float("-inf")
            if not do_sample or temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k and 0 < top_k < logits.size(-1):
                    kth = torch.topk(logits, top_k).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p and top_p < 1.0:
                    srt, idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1) > top_p
                    cum[..., 1:] = cum[..., :-1].clone()
                    cum[..., 0] = False
                    logits = logits.masked_fill(cum.scatter(1, idx, cum), float("-inf"))
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
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


def cfg_diffq(**kw):
    d = int(kw.pop("hidden_size", 256))
    attn_heads = int(kw.pop("num_attention_heads", 8))
    L = int(kw.pop("num_hidden_layers", 10))
    cfg = {
        "arch_type": "std_gqa_diff_qpool",
        "vocab_size": int(kw.pop("vocab_size", 8302)),
        "hidden_size": d,
        "num_hidden_layers": L,
        "q_pool_size": int(kw.pop("q_pool_size", L + 1)),
        "num_attention_heads": attn_heads,
        "num_key_value_heads": int(kw.pop("num_key_value_heads", 2)),
        "head_dim": int(kw.pop("head_dim", d // attn_heads)),
        "intermediate_size": int(kw.pop("intermediate_size", 1072)),
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


def param_report_diffq(model=None, cfg=None, verbose=True):
    cfg = dict(cfg or (model.cfg if model is not None else {}))
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    q_pool = int(cfg.get("q_pool_size") or (L + 1))
    h_q, h_kv, hd = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
    inter = int(cfg["intermediate_size"])

    embed = int(cfg["vocab_size"]) * d * (1 if cfg.get("tie_word_embeddings", True) else 2)
    q_pool_params = q_pool * d * (h_q * hd)
    kvo_per = 2 * d * (h_kv * hd) + (h_q * hd) * d + hd                 # W_K + W_V + W_O + k_norm
    diff_extra_per = 3 * (h_q * hd) + 4 * hd                            # q1_norm + q2_norm + sub_norm + 4*lambda
    attn_total = q_pool_params + L * (kvo_per + diff_extra_per)
    ffn_per = 3 * d * inter
    norms_total = L * (2 * d) + d
    total = embed + attn_total + L * ffn_per + norms_total

    measured = model.num_params() if model is not None else None
    if verbose:
        print("=" * 84)
        print("相邻双Q共享池 + 差分去噪(Diff-Attn) + SwiGLU  d=%d ｜ %d层 ｜ Q池=%d ｜ %dQ×2 / %dKV (hd=%d) ｜ i=%d (%.2f×)"
              % (d, L, q_pool, h_q, h_kv, hd, inter, inter / float(d)))
        print("  占比：Embed %.2fM(%.1f%%) ｜ Attn+Q池 %.2fM(%.1f%%) ｜ FFN %.2fM(%.1f%%)"
              % (embed / 1e6, 100.0 * embed / total,
                 attn_total / 1e6, 100.0 * attn_total / total,
                 (L * ffn_per) / 1e6, 100.0 * (L * ffn_per) / total))
        print("  公式总数 %s (%.4f M) ｜ 4bit ≈ %.2f MiB ｜ bf16 ≈ %.2f MiB"
              % ("{:,}".format(total), total / 1e6, total * 0.5 / 1048576, total * 2.0 / 1048576))
        if measured is not None:
            print("  torch 实测 %s (%.4f M) ｜ 差异 %d"
                  % ("{:,}".format(measured), measured / 1e6, measured - total))
            assert measured == total, "公式 %d != 实测 %d" % (total, measured)
        print("=" * 84)
    return measured, total
