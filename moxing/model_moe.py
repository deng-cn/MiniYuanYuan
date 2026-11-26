# -*- coding: utf-8 -*-
"""model_moe.py —— 1125 5号放大版（12层）DeepSeek-MoE 稀疏热升级架构

在原有 12 层（每 2 层共享 1 组 BiCrossGateFFN，共 6 组）基础上：
  · 保留 Group 0, 1, 2（L0-L5）与 Group 5（L10-L11）为稠密共享 BiCrossGateFFN (i=1152)
  · 将中深层事实记忆核心区 Group 3（L6-L7，倒数第3组）和 Group 4（L8-L9，倒数第2组）
    升级为 DeepSeek-MoE 混合专家池：
      1) 1 个常驻共享专家（Shared Expert, i=1152）：100% 继承第 2 轮收敛权重，保底零遗忘
      2) E=6 个细粒度路由专家（每个 BiCrossGateFFN, i=384），Top-2 激活（C(6,2)=15 种组合）
      3) 共享组内每层配独立路由器（L6/L7 独立路由，L8/L9 独立路由），用 sigmoid 门控 + 归一化 Top-K 权重
      4) DeepSeek-V3 无辅助损失动态偏置均衡（Aux-Loss-Free Dynamic Bias）+ 极轻序列均衡正则（alpha=5e-4）
      5) GPU 零同步/单同步高效分发（训练走 argsort + 单次 bincount；单 token 生成走稀疏直调）
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
from model_p4 import BiCrossGateFFN


class MoERouter(nn.Module):
    """Per-layer DeepSeek-V3 风格路由器（sigmoid 亲和度 + 无辅助损失偏置均衡）。"""

    def __init__(self, d, num_experts, top_k=2, bias_lr=0.001, bias_clamp=1.0, seq_aux_alpha=5e-4):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.bias_lr = float(bias_lr)
        self.bias_clamp = float(bias_clamp)
        self.seq_aux_alpha = float(seq_aux_alpha)
        # 命名刻意避开 gate_proj / _LINEAR_LEAVES，避免 4bit 量化误伤路由器
        self.proj = nn.Linear(d, self.num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("_accum_counts", torch.zeros(self.num_experts, dtype=torch.float64), persistent=False)
        self.register_buffer("_accum_tokens", torch.zeros(1, dtype=torch.float64), persistent=False)
        self._collect_stats = False
        self._stat_counts = None
        self._stat_tokens = 0

    def enable_stats(self, flag=True):
        self._collect_stats = bool(flag)
        if flag:
            self._stat_counts = torch.zeros(self.num_experts, dtype=torch.float64, device=self.proj.weight.device)
            self._stat_tokens = 0

    def pop_stats(self):
        self._collect_stats = False
        if self._stat_counts is None or self._stat_tokens == 0:
            return None
        probs = (self._stat_counts / float(self._stat_tokens * self.top_k)).detach().cpu().tolist()
        p_t = torch.tensor(probs, dtype=torch.float64).clamp_min(1e-12)
        ent = float(-(p_t * p_t.log()).sum() / math.log(max(2, self.num_experts)))
        self._stat_counts = None
        self._stat_tokens = 0
        return {
            "probs": probs,
            "norm_entropy": ent,
            "min_p": min(probs),
            "max_p": max(probs),
            "bias": [round(float(x), 3) for x in self.expert_bias.detach().cpu().tolist()],
        }

    def forward(self, x):
        """x: (B, T, d) -> (topk_idx (B*T, K), topk_w (B*T, K), aux_loss)"""
        B, T, D = x.shape
        logits = self.proj(x).float()                           # (B, T, E)
        scores = torch.sigmoid(logits)                          # DeepSeek-V3 sigmoid affinity
        sel_scores = scores + self.expert_bias.view(1, 1, -1)
        topk_idx = torch.topk(sel_scores, k=self.top_k, dim=-1).indices  # (B, T, K)
        raw_w = torch.gather(scores, -1, topk_idx)              # unbiased weights
        topk_w = raw_w / (raw_w.sum(dim=-1, keepdim=True) + 1e-6)

        flat_idx = topk_idx.reshape(-1, self.top_k)
        flat_w = topk_w.reshape(-1, self.top_k).to(x.dtype)

        if self.training:
            with torch.no_grad():
                cnt = torch.bincount(flat_idx.reshape(-1), minlength=self.num_experts).to(torch.float64)
                self._accum_counts.add_(cnt)
                self._accum_tokens.add_(float(B * T * self.top_k))
        if self._collect_stats:
            with torch.no_grad():
                cnt = torch.bincount(flat_idx.reshape(-1), minlength=self.num_experts).to(torch.float64)
                self._stat_counts.add_(cnt)
                self._stat_tokens += int(B * T)

        aux_loss = None
        if self.training and self.seq_aux_alpha > 0.0:
            # DeepSeek-V3 sequence-wise auxiliary loss（极小系数，防单条序列极端坍塌）
            p_seq = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)   # (B, T, E)
            one_hot = F.one_hot(topk_idx, num_classes=self.num_experts).float().sum(dim=-2)  # (B, T, E)
            f_seq = one_hot.mean(dim=1) * (float(self.num_experts) / float(self.top_k))      # (B, E)
            p_mean = p_seq.mean(dim=1)                                                       # (B, E)
            aux_loss = self.seq_aux_alpha * (f_seq * p_mean).sum(dim=-1).mean()

        return flat_idx, flat_w, aux_loss

    @torch.no_grad()
    def update_bias(self):
        tot = float(self._accum_tokens.item())
        if tot <= 0.0:
            return
        freq = (self._accum_counts / tot).to(self.expert_bias.dtype)
        target = 1.0 / float(self.num_experts)
        delta = self.bias_lr * torch.sign(target - freq)
        self.expert_bias.add_(delta).clamp_(-self.bias_clamp, self.bias_clamp)
        self._accum_counts.zero_()
        self._accum_tokens.zero_()


class MoEBiCrossGatePool(nn.Module):
    """共享 BiCrossGate 专家 + E 个细粒度路由 BiCrossGate 专家（不含路由器，路由器按层独立）。"""

    def __init__(self, cfg):
        super().__init__()
        self.shared = BiCrossGateFFN(cfg)
        routed_cfg = dict(cfg)
        routed_cfg["intermediate_size"] = int(cfg["moe_intermediate_size"])
        self.num_experts = int(cfg["num_experts"])
        self.experts = nn.ModuleList([BiCrossGateFFN(routed_cfg) for _ in range(self.num_experts)])

    def forward_with_router(self, x, router: MoERouter):
        """x: (B, T, D) -> (out (B, T, D), aux_loss)"""
        B, T, D = x.shape
        N = B * T
        shared_out = self.shared(x)
        topk_idx, topk_w, aux_loss = router(x)  # (N, K), (N, K)
        x_flat = x.reshape(N, D)
        K = topk_idx.shape[1]

        # 推理小步解码（B*T <= 4）直接按选中专家稀疏调用，零 argsort/bincount 开销
        if not self.training and N <= 4:
            routed = torch.zeros_like(x_flat)
            for ti in range(N):
                xt = x_flat[ti:ti + 1]
                acc = torch.zeros_like(xt)
                for ki in range(K):
                    eid = int(topk_idx[ti, ki].item())
                    acc = acc + self.experts[eid](xt) * topk_w[ti, ki]
                routed[ti:ti + 1] = acc
            return shared_out + routed.reshape(B, T, D), aux_loss

        # 批处理/训练：按专家排序后用单次 bincount 切块，避免逐专家 nonzero() 的 E 次 CUDA 同步
        flat_e = topk_idx.reshape(-1)
        flat_w = topk_w.reshape(-1)
        order = torch.argsort(flat_e, stable=False)
        sorted_e = flat_e[order]
        counts = torch.bincount(sorted_e, minlength=self.num_experts).tolist()
        tok_order = torch.div(order, K, rounding_mode="floor")
        x_sorted = x_flat[tok_order]
        w_sorted = flat_w[order].unsqueeze(-1)
        chunks = torch.split(x_sorted, counts, dim=0)
        ys = [self.experts[e](chunks[e]) for e in range(self.num_experts) if counts[e] > 0]
        if not ys:
            return shared_out, aux_loss
        y_cat = torch.cat(ys, dim=0) * w_sorted
        routed = torch.zeros_like(x_flat)
        routed.index_add_(0, tok_order, y_cat.to(routed.dtype))
        return shared_out + routed.reshape(B, T, D), aux_loss


class MoEBlock(nn.Module):
    def __init__(self, cfg, idx=0, shared_ffn=False, is_moe=False):
        super().__init__()
        d = int(cfg["hidden_size"])
        eps = float(cfg["rms_norm_eps"])
        self.idx = idx
        self.is_moe = bool(is_moe)
        self.input_layernorm = RMSNorm(d, eps=eps)
        self.self_attn = StandardGQAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.ffn = None if shared_ffn else (MoEBiCrossGatePool(cfg) if self.is_moe else BiCrossGateFFN(cfg))
        self.moe_router = MoERouter(
            d,
            num_experts=int(cfg["num_experts"]),
            top_k=int(cfg["moe_top_k"]),
            bias_lr=float(cfg.get("moe_bias_lr", 0.001)),
            bias_clamp=float(cfg.get("moe_bias_clamp", 1.0)),
            seq_aux_alpha=float(cfg.get("moe_seq_aux_alpha", 5e-4)),
        ) if self.is_moe else None

    def forward(self, hidden_states, q_rope, k_rope, past_kv=None, use_cache=False,
                attention_mask=None, ffn_override=None):
        x_norm = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            x_norm, q_rope, k_rope, past_kv=past_kv, use_cache=use_cache, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        mlp_in = self.post_attention_layernorm(hidden_states)
        mlp = ffn_override if ffn_override is not None else self.ffn
        aux_loss = None
        if self.is_moe:
            ffn_out, aux_loss = mlp.forward_with_router(mlp_in, self.moe_router)
        else:
            ffn_out = mlp(mlp_in)
        hidden_states = hidden_states + ffn_out
        return hidden_states, present, aux_loss


class MoEBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers = int(cfg["num_hidden_layers"])
        self.ffn_share = int(cfg.get("ffn_share_group") or 1)
        self.moe_groups = set(int(g) for g in (cfg.get("moe_groups") or [3, 4]))
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        d = int(cfg["hidden_size"])
        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), d)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))
        if self.ffn_share > 1:
            n_groups = (self.n_layers + self.ffn_share - 1) // self.ffn_share
            self.ffn_groups = nn.ModuleList([
                MoEBiCrossGatePool(cfg) if g in self.moe_groups else BiCrossGateFFN(cfg)
                for g in range(n_groups)
            ])
            self.layers = nn.ModuleList([
                MoEBlock(cfg, i, shared_ffn=True, is_moe=((i // self.ffn_share) in self.moe_groups))
                for i in range(self.n_layers)
            ])
        else:
            self.ffn_groups = None
            self.layers = nn.ModuleList([
                MoEBlock(cfg, i, shared_ffn=False, is_moe=(i in self.moe_groups))
                for i in range(self.n_layers)
            ])
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
        aux_losses = []
        if self.grad_checkpoint and self.training and not use_cache and past_key_values is None:
            for i, layer in enumerate(self.layers):
                ffn_ov = self.ffn_groups[i // self.ffn_share] if self.ffn_groups is not None else None
                def _fn(h, _lay=layer, _fo=ffn_ov):
                    out_h, _, al = _lay(h, q_rope, k_rope, None, False, attention_mask, ffn_override=_fo)
                    if al is None:
                        al = out_h.new_zeros((), dtype=torch.float32)
                    return out_h, al
                hidden, al = _ckpt(_fn, hidden, use_reentrant=False)
                if layer.is_moe:
                    aux_losses.append(al)
            aux_total = sum(aux_losses) if aux_losses else None
            return self.norm(hidden), [], aux_total

        presents = []
        for i, (layer, pkv) in enumerate(zip(self.layers, past)):
            ffn_ov = self.ffn_groups[i // self.ffn_share] if self.ffn_groups is not None else None
            hidden, pres, al = layer(hidden, q_rope, k_rope, past_kv=pkv,
                                     use_cache=use_cache, attention_mask=attention_mask,
                                     ffn_override=ffn_ov)
            if use_cache:
                presents.append(pres)
            if al is not None:
                aux_losses.append(al)
        aux_total = sum(aux_losses) if aux_losses else None
        return self.norm(hidden), presents, aux_total


class LMMoE(nn.Module):
    """DeepSeek-MoE 稀疏热升级版 5 号模型。"""

    def __init__(self, cfg):
        super().__init__()
        cfg = cfg_moe(**cfg)
        self.cfg = cfg
        self.model = MoEBackbone(cfg)
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
                if isinstance(layer.ffn, BiCrossGateFFN):
                    nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=ffn_std)
                elif isinstance(layer.ffn, MoEBiCrossGatePool):
                    nn.init.normal_(layer.ffn.shared.down_proj.weight, mean=0.0, std=ffn_std)
                if layer.moe_router is not None:
                    nn.init.normal_(layer.moe_router.proj.weight, mean=0.0, std=0.01)
            if self.model.ffn_groups is not None:
                for grp in self.model.ffn_groups:
                    if isinstance(grp, BiCrossGateFFN):
                        nn.init.normal_(grp.down_proj.weight, mean=0.0, std=ffn_std)
                    elif isinstance(grp, MoEBiCrossGatePool):
                        nn.init.normal_(grp.shared.down_proj.weight, mean=0.0, std=ffn_std)

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

    def enable_moe_stats(self):
        for layer in self.model.layers:
            if layer.moe_router is not None:
                layer.moe_router.enable_stats(True)

    def pop_moe_stats(self):
        out = []
        for layer in self.model.layers:
            if layer.moe_router is not None:
                out.append((layer.idx, layer.moe_router.pop_stats()))
        return out

    @torch.no_grad()
    def update_moe_bias(self):
        for layer in self.model.layers:
            if layer.moe_router is not None:
                layer.moe_router.update_bias()

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kw):
        hidden, presents, aux_loss = self.model(
            input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, position_offset=kw.get("position_offset"))
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep else slice(None)
        logits = self.lm_head(hidden[:, sl, :])
        ce_loss = None
        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            if (y != IGNORE).any():
                ce_loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=IGNORE)
            else:
                ce_loss = logits.sum() * 0.0
            loss = ce_loss if aux_loss is None else (ce_loss + aux_loss.to(ce_loss.dtype))
        return {
            "loss": loss, "ce_loss": ce_loss, "aux_loss": aux_loss,
            "logits": logits, "past_key_values": presents,
            "hidden_states": hidden, "mtp_loss": None,
        }

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


@torch.no_grad()
def upcycle_from_p4_state(model: LMMoE, p4_sd: dict, down_scale=0.05, q_noise_ratio=0.10, seed=42):
    """DeepSeek-MoE 稀疏热升级（Sparse Upcycling）：
    1) Embedding / Norm / 全层 Attention / 非 MoE FFN 组：100% 精确拷贝；
    2) MoE 组：
       - shared：100% 精确拷贝原 BiCrossGateFFN(i=1152)，保底不掉能力；
       - experts[e] (i=384)：前半通道 (shared_i // E = 192) 从原 FFN 切分继承（错位分工），
         后半通道随机初始化补充新容量；down_proj 继承段缩放到 down_scale，新段 1e-3 小初始化；
       - router：std=0.01 微扰初始化，expert_bias=0，从第 1 步即提供可微有效梯度。
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    dst_sd = model.state_dict()
    copied_direct = 0

    # 1. 直接匹配拷贝
    for k, v in p4_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k].copy_(v.to(dst_sd[k].device, dtype=dst_sd[k].dtype))
            copied_direct += 1

    moe_groups = sorted(model.model.moe_groups)
    E = int(model.cfg["num_experts"])
    shared_i = int(model.cfg["intermediate_size"])
    routed_i = int(model.cfg["moe_intermediate_size"])
    slice_w = min(routed_i, shared_i // E)  # 1152 // 6 = 192

    for g in moe_groups:
        src_pref = f"model.ffn_groups.{g}."
        dst_pref = f"model.ffn_groups.{g}."
        # 2a. 拷贝 shared 专家
        for sub in ("q.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            sk = src_pref + sub
            dk = dst_pref + "shared." + sub
            if sk not in p4_sd:
                raise KeyError(f"原权重缺少 {sk}")
            dst_sd[dk].copy_(p4_sd[sk].to(dst_sd[dk].device, dtype=dst_sd[dk].dtype))

        wq = p4_sd[src_pref + "q.weight"].float()
        wg = p4_sd[src_pref + "gate_proj.weight"].float()   # (1152, 2d)
        wu = p4_sd[src_pref + "up_proj.weight"].float()     # (1152, d)
        wd = p4_sd[src_pref + "down_proj.weight"].float()   # (d, 1152)
        q_std = float(wq.std().item()) * float(q_noise_ratio)

        # 2b. 切分继承 + 新容量初始化 E 个路由专家
        for e in range(E):
            ep = f"{dst_pref}experts.{e}."
            # q: 继承原双线性特征 + 微小分工扰动
            qn = torch.randn(wq.shape, generator=gen, dtype=torch.float32) * q_std
            dst_sd[ep + "q.weight"].copy_((wq + qn).to(dst_sd[ep + "q.weight"].device, dtype=dst_sd[ep + "q.weight"].dtype))

            s0, s1 = e * slice_w, (e + 1) * slice_w
            # gate_proj: (routed_i, 2d)
            g_new = torch.randn(dst_sd[ep + "gate_proj.weight"].shape, generator=gen, dtype=torch.float32) * 0.02
            g_new[:slice_w, :] = wg[s0:s1, :]
            dst_sd[ep + "gate_proj.weight"].copy_(g_new.to(dst_sd[ep + "gate_proj.weight"].device, dtype=dst_sd[ep + "gate_proj.weight"].dtype))

            # up_proj: (routed_i, d)
            u_new = torch.randn(dst_sd[ep + "up_proj.weight"].shape, generator=gen, dtype=torch.float32) * 0.02
            u_new[:slice_w, :] = wu[s0:s1, :]
            dst_sd[ep + "up_proj.weight"].copy_(u_new.to(dst_sd[ep + "up_proj.weight"].device, dtype=dst_sd[ep + "up_proj.weight"].dtype))

            # down_proj: (d, routed_i) — 小比例起步，防第 0 步冲击主干同时保证路由器从第 1 步起就有梯度
            d_new = torch.randn(dst_sd[ep + "down_proj.weight"].shape, generator=gen, dtype=torch.float32) * 1e-3
            d_new[:, :slice_w] = wd[:, s0:s1] * float(down_scale)
            dst_sd[ep + "down_proj.weight"].copy_(d_new.to(dst_sd[ep + "down_proj.weight"].device, dtype=dst_sd[ep + "down_proj.weight"].dtype))

    model.load_state_dict(dst_sd)
    return {"copied_direct": copied_direct, "moe_groups": moe_groups, "slice_w": slice_w, "experts": E}


def cfg_moe(**kw):
    d = int(kw.pop("hidden_size", 384))
    attn_heads = int(kw.pop("num_attention_heads", 12))
    cfg = {
        "arch_type": "std_gqa_bicross_moe",
        "vocab_size": int(kw.pop("vocab_size", 8302)),
        "hidden_size": d,
        "num_hidden_layers": int(kw.pop("num_hidden_layers", 12)),
        "num_attention_heads": attn_heads,
        "num_key_value_heads": int(kw.pop("num_key_value_heads", 2)),
        "head_dim": int(kw.pop("head_dim", d // attn_heads)),
        "intermediate_size": int(kw.pop("intermediate_size", 1152)),
        "ffn_share_group": int(kw.pop("ffn_share_group", 2) or 1),
        "moe_groups": list(kw.pop("moe_groups", [3, 4])),
        "num_experts": int(kw.pop("num_experts", 6)),
        "moe_top_k": int(kw.pop("moe_top_k", 2)),
        "moe_intermediate_size": int(kw.pop("moe_intermediate_size", d)),
        "moe_bias_lr": float(kw.pop("moe_bias_lr", 0.001)),
        "moe_bias_clamp": float(kw.pop("moe_bias_clamp", 1.0)),
        "moe_seq_aux_alpha": float(kw.pop("moe_seq_aux_alpha", 5e-4)),
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


def param_report_moe(model=None, cfg=None, verbose=True):
    cfg = dict(cfg or (model.cfg if model is not None else {}))
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    share = int(cfg.get("ffn_share_group") or 1)
    n_groups = (L + share - 1) // share if share > 1 else L
    moe_groups = set(int(g) for g in (cfg.get("moe_groups") or []))
    n_moe_g = len(moe_groups)
    n_moe_l = sum(1 for i in range(L) if ((i // share) if share > 1 else i) in moe_groups)
    E = int(cfg["num_experts"])
    K = int(cfg["moe_top_k"])
    ri = int(cfg["moe_intermediate_size"])
    h_q, h_kv, hd = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
    inter = int(cfg["intermediate_size"])
    embed = int(cfg["vocab_size"]) * d * (1 if cfg.get("tie_word_embeddings", True) else 2)
    attn_per = d * (h_q * hd) + 2 * d * (h_kv * hd) + (h_q * hd) * d + 2 * hd
    ffn_per = 2 * d * d + 4 * d * inter
    exp_per = 2 * d * d + 4 * d * ri
    router_per = d * E
    norms_per = 2 * d
    base_total = embed + L * (attn_per + norms_per) + n_groups * ffn_per + d
    moe_added = n_moe_g * E * exp_per + n_moe_l * router_per
    total = base_total + moe_added
    # 推理激活等效参数（按 MoE 组唯一权重 + 每组激活 K 个专家估算）
    active_unique = base_total + n_moe_g * K * exp_per + n_moe_l * router_per
    measured = model.num_params() if model is not None else None
    if verbose:
        print("=" * 84)
        print("5号升级版 DeepSeek-MoE（%d组FFN中第%s组升为MoE: 1共享+%d专家Top-%d, i_exp=%d）d=%d ｜ %d层"
              % (n_groups, sorted(moe_groups), E, K, ri, d, L))
        print("  基座(19.71M) + 新增MoE %.2fM（%d组×%d专家 + %d层独立路由）"
              % (moe_added / 1e6, n_moe_g, E, n_moe_l))
        print("  总参 %s (%.2f M) ｜ 4bit ≈ %.2f MiB ｜ 单token激活等效 ≈ %.2f M"
              % ("{:,}".format(total), total / 1e6, total * 0.5 / 1048576, active_unique / 1e6))
        if measured is not None:
            print("  torch 实测 %s (%.4f M) ｜ 差异 %d"
                  % ("{:,}".format(measured), measured / 1e6, measured - total))
            assert measured == total, "公式 %d != 实测 %d" % (total, measured)
        print("=" * 84)
    return measured, total
