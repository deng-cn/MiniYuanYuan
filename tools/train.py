# -*- coding: utf-8 -*-
"""train.py —— 1126 架构（10层独立 4Q-NoKV + BiCross + SwiGLU 2d）训练器"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "moxing"))

from data import PackedDataset, repack                   # noqa: E402
from tokenizer import Tokenizer, default_vocab_path        # noqa: E402
from model_1126 import LM1126, cfg_1126, param_report_1126 # noqa: E402

PROMPTS = ["苹果是什么？", "因为下雨了，所以", "你好，你是谁？"]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class Ctx(object):
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        self.metrics = os.path.join(run_dir, "metrics.jsonl")
        self.status = os.path.join(run_dir, "status.json")
        self.logf = os.path.join(run_dir, "log.txt")

    def log(self, msg):
        line = "[%s] %s" % (now(), msg)
        print(line, flush=True)
        with open(self.logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def metric(self, rec):
        with open(self.metrics, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def heartbeat(self, rec):
        atomic_json(self.status, rec)


def lr_at(step, total, base, warmup, min_ratio=0.05):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return base * (min_ratio + (1 - min_ratio) * cos)


def save_ckpt(path, model, opt, step, epoch, best_val, cfg, args, keep_opt=True):
    obj = {
        "model": model.state_dict(),
        "step": step, "epoch": epoch, "best_val": best_val, "cfg": cfg,
        "args": vars(args), "saved_at": now(),
    }
    if keep_opt and opt is not None:
        obj["opt"] = opt.state_dict()
        obj["rng"] = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    torch.save(obj, path)


def pure_ce(out):
    loss = out["loss"]
    if out.get("mtp_loss") is not None:
        loss = loss - out["mtp_loss"]
    return loss


@torch.no_grad()
def evaluate(model, ds, batch_size, device, dtype, max_batches=40, collect_gate=True):
    model.eval()
    if collect_gate:
        model.enable_gate_stats()
    idx = np.arange(len(ds))
    if len(idx) > batch_size * max_batches:
        idx = np.random.RandomState(0).choice(idx, batch_size * max_batches, replace=False)
    losses, ntok = 0.0, 0
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        x, y = ds.get_batch(b, device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(x, labels=y)
        n = int((y[:, 1:] != -100).sum())
        losses += float(pure_ce(out)) * n
        ntok += n
    gstats = model.pop_gate_stats() if collect_gate else None
    model.train()
    return losses / max(1, ntok), gstats


def summarize_gate_stats(gstats):
    if not gstats or not gstats[0]:
        return ""
    ents = [s["norm_entropy"] for s in gstats if s]
    off_cos = []
    for s in gstats:
        if not s:
            continue
        c = s["pairwise_cos"]
        m = len(c)
        vals = [c[i][j] for i in range(m) for j in range(m) if i != j]
        if vals:
            off_cos.append(sum(vals) / len(vals))
    return "gate_H=%.3f, q_cos=%.3f" % (
        sum(ents) / max(1, len(ents)),
        sum(off_cos) / max(1, len(off_cos)),
    )


@torch.no_grad()
def sample(model, tk, device, amp_dtype=torch.bfloat16, max_new=64):
    model.eval()
    use_amp = (device == "cuda") and (amp_dtype != torch.float32)
    outs = []
    for p in PROMPTS:
        text = "<bos><user>" + p + "<sep><assistant>"
        ids = torch.tensor([tk.encode(text)], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            r = model.generate(ids, max_new_tokens=max_new, temperature=0.6, top_p=0.9, top_k=50,
                               eos_token_id=[tk.eos_id, tk.stoi["<sep>"]], repetition_penalty=1.15,
                               no_repeat_ngram_size=3, context_len=ids.shape[1])
        outs.append("%s\n-> %s" % (p, tk.decode(r["new_tokens"], skip_special_tokens=True)))
    model.train()
    return "\n\n".join(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=default_vocab_path())
    ap.add_argument("--data", default=r"D:\AI_xunlian\1124\dataset\_mix\packed_v2")
    ap.add_argument("--run", default="1126_d384_l10_4q_dualffn")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--config",
                    default=os.path.join(ROOT, "configs", "d384_l10_4q_nokv_bicross_swiglu2d_w1088.json"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1088)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--grad-ckpt", default="none", choices=["none", "layer"])
    ap.add_argument("--attn-backend", default="cudnn",
                    choices=["auto", "cudnn", "efficient", "math", "flash"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--q-group", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--keep-ckpt", type=int, default=2)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--repack-cache", default="")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    run_dir = os.path.join(args.out, args.run)
    ctx = Ctx(run_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    atomic_json(os.path.join(run_dir, "run_meta.json"),
                {"args": vars(args), "started": now(), "pid": os.getpid(),
                 "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0)
                 if torch.cuda.is_available() else None})

    if not os.path.isfile(args.vocab):
        raise SystemExit("找不到词表：%s" % args.vocab)
    if not os.path.isdir(args.data):
        raise SystemExit("找不到数据集目录：%s" % args.data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu":
        amp_dtype = torch.float32
    ctx.log("设备 %s ｜ dtype=%s ｜ run=%s ｜ pid=%d ｜ grad_ckpt=%s"
            % (device, args.dtype, args.run, os.getpid(), args.grad_ckpt))

    with open(args.vocab, encoding="utf-8") as f:
        vd = json.load(f)
    with open(args.config, encoding="utf-8") as f:
        raw = json.load(f)
    raw.pop("_说明", None)
    raw["vocab_size"] = vd["total"]
    raw["max_position_embeddings"] = max(int(raw.get("max_position_embeddings", 1088)), args.max_len)
    raw["eos_token_id"] = vd.get("model_cfg", {}).get("eos_token_id", 3)
    raw["attn_backend"] = args.attn_backend
    raw["grad_checkpoint"] = (args.grad_ckpt != "none")
    cfg = cfg_1126(**raw)

    ctx.log("配置 %s：d=%d ｜ %d层独立 ｜ %d套Q×%d头(hd%d) ｜ BiCross+SwiGLU(%d) ｜ MTP=%d"
            % (os.path.basename(args.config), cfg["hidden_size"], cfg["num_hidden_layers"],
               cfg["num_q_sets"], cfg["num_key_value_heads"], cfg["head_dim"],
               cfg["intermediate_size"], int(cfg["mtp"])))
    tk = Tokenizer(vd)

    # ---- 数据 ----
    tr = PackedDataset(args.data, "train", args.max_len)
    va = PackedDataset(args.data, "val", args.max_len)
    if len(tr) == 0:
        raise SystemExit("train.bin 为空或 max_len(%d) 与打包时不一致" % args.max_len)
    accum = max(1, args.grad_accum)
    steps_per_epoch = max(1, len(tr) // (args.batch * accum))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    tokens_per_step = args.batch * accum * args.max_len
    ctx.log("数据 train=%d 窗口 val=%d 窗口 ｜ 每步 %d 窗 = %d token"
            % (len(tr), len(va), args.batch * accum, tokens_per_step))
    ctx.log("steps/epoch=%d ｜ 计划 %d 步（%d epoch）" % (steps_per_epoch, total_steps, args.epochs))

    # ---- 模型 ----
    model = LM1126(cfg).to(device)
    param_report_1126(model, cfg)
    if args.qat:
        from quant import apply_qat
        _qinfo = apply_qat(model, group=args.q_group)
        ctx.log("QAT 已启用：%s" % _qinfo)
    nparam = model.num_params()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim == 1 or "norm" in n) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # ---- 续训 / 热启动 ----
    start_step, start_epoch, best_val = 0, 0, float("inf")
    last_path = os.path.join(run_dir, "last.pt")
    if args.init_weights:
        ck = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ctx.log("热启动：载入 %s（来源 step=%s best_val=%.4f）；优化器与步数清零"
                % (args.init_weights, ck.get("step"), ck.get("best_val", float("inf"))))
    elif args.resume == "auto" and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        start_step, start_epoch, best_val = ck["step"], ck["epoch"], ck.get("best_val", float("inf"))
        if ck.get("rng"):
            try:
                torch.set_rng_state(ck["rng"]["torch"].cpu() if torch.is_tensor(ck["rng"]["torch"])
                                    else ck["rng"]["torch"])
                np.random.set_state(ck["rng"]["numpy"])
            except Exception:
                pass
        ctx.log("续训：从 step=%d epoch=%d best_val=%.4f 继续" % (start_step, start_epoch, best_val))
    elif args.resume not in ("auto", "none", ""):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        start_step, start_epoch, best_val = ck["step"], ck["epoch"], ck.get("best_val", float("inf"))
        if ck.get("rng"):
            try:
                torch.set_rng_state(ck["rng"]["torch"].cpu() if torch.is_tensor(ck["rng"]["torch"])
                                    else ck["rng"]["torch"])
                np.random.set_state(ck["rng"]["numpy"])
            except Exception:
                pass
        ctx.log("续训：从 %s（step=%d epoch=%d best_val=%.4f，含优化器:%s）"
                % (args.resume, start_step, start_epoch, best_val, "是" if ck.get("opt") is not None else "否"))

    stop_flag = {"stop": False}

    def handler(signum, frame):
        stop_flag["stop"] = True
        ctx.log("收到信号 %d —— 保存后退出" % signum)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, handler)
        except Exception:
            pass

    # ---- 训练循环 ----
    model.train()
    step = start_step
    epoch = start_epoch
    t_start = time.time()
    tok_seen = 0
    run_loss, run_n = 0.0, 0
    grad_norm = 0.0
    use_amp = (amp_dtype != torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    rep_round = start_step // steps_per_epoch
    while step < total_steps and not stop_flag["stop"]:
        epoch = rep_round
        if args.repack_cache:
            seed = args.seed + rep_round * 10007
            repack(args.repack_cache, args.data, seed=seed,
                   max_len=args.max_len, quiet=True)
            tr.reload()
            va.reload()
            ctx.log("epoch %d 重新打散数据（seed=%d）：train %d 窗 / val %d 窗"
                    % (epoch, seed, len(tr), len(va)))
        rep_round += 1
        order = np.random.permutation(len(tr))
        n_batches = max(1, len(tr) // (args.batch * accum))
        for bi in range(n_batches):
            if step >= total_steps or stop_flag["stop"]:
                break
            i0 = bi * args.batch * accum
            idx = order[i0:i0 + args.batch * accum]
            if len(idx) < args.batch:
                continue
            lr = lr_at(step, total_steps, args.lr, args.warmup, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr

            step_loss, step_tok = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for mi in range(accum):
                sub = idx[mi * args.batch:(mi + 1) * args.batch]
                if len(sub) < 1:
                    break
                x, y = tr.get_batch(sub, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(x, labels=y)
                    loss = out["loss"]
                scaler.scale(loss / accum).backward()
                step_loss += float(pure_ce(out).detach())
                step_tok += int(x.numel())

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            step += 1
            tok_seen += step_tok
            run_loss += step_loss / max(1, accum)
            run_n += 1

            if step % args.log_every == 0:
                avg = run_loss / max(1, run_n)
                el = time.time() - t_start
                tps = tok_seen / max(1e-9, el)
                eta = (total_steps - step) / max(1e-9, (step - start_step) / max(1e-9, el))
                mem = torch.cuda.memory_allocated() / 1048576 if device == "cuda" else 0
                peak = torch.cuda.max_memory_allocated() / 1048576 if device == "cuda" else 0
                rec = {"step": step, "epoch": epoch, "loss": round(avg, 4), "lr": lr,
                       "grad_norm": round(grad_norm, 3), "tok_s": int(tps),
                       "gpu_mem_mb": round(mem, 0), "gpu_peak_mb": round(peak, 0), "t": now()}
                ctx.metric(rec)
                ctx.heartbeat({"pid": os.getpid(), "run": args.run, "step": step,
                               "total_steps": total_steps, "epoch": epoch,
                               "epochs": args.epochs, "loss": round(avg, 4),
                               "best_val": None if best_val == float("inf") else round(best_val, 4),
                               "lr": lr, "tok_s": int(tps), "eta_min": round(eta / 60, 1),
                               "grad_norm": round(grad_norm, 3), "gpu_mem_mb": round(mem, 0),
                               "gpu_peak_mb": round(peak, 0), "updated": now(),
                               "params": nparam, "tokens_per_step": tokens_per_step,
                               "grad_ckpt": args.grad_ckpt})
                ctx.log("step %d/%d epoch %d ｜ loss %.4f ｜ lr %.2e ｜ %.0f tok/s ｜ grad %.2f ｜ mem %.0f/%.0fMB ｜ ETA %.0f min"
                        % (step, total_steps, epoch, avg, lr, tps, grad_norm, mem, peak, eta / 60))
                run_loss, run_n = 0.0, 0

            if step % args.val_every == 0 and len(va) > 0:
                vl, gstats = evaluate(model, va, args.batch, device, amp_dtype)
                gs_txt = summarize_gate_stats(gstats)
                layer_diag = [
                    {"l": li,
                     "H": round(s["norm_entropy"], 4),
                     "p": [round(v, 3) for v in s["mean_q_probs"]]}
                    for li, s in enumerate(gstats or []) if s
                ]
                ctx.metric({"step": step, "val_loss": round(vl, 4),
                            "gate_summary": gs_txt, "gate_layers": layer_diag, "t": now()})
                ctx.log("  ★ val_loss %.4f（best %.4f）｜ %s" % (vl, best_val, gs_txt))
                if vl < best_val:
                    best_val = vl
                    save_ckpt(os.path.join(run_dir, "best.pt"), model, None, step, epoch,
                              best_val, cfg, args, keep_opt=False)
                    ctx.log("  ★ 新最优，已存 best.pt")

            if step % args.sample_every == 0:
                txt = sample(model, tk, device, amp_dtype=amp_dtype)
                with open(os.path.join(run_dir, "samples", "step_%06d.txt" % step),
                          "w", encoding="utf-8") as f:
                    f.write("[%s] step %d\n\n%s\n" % (now(), step, txt))
                ctx.log("  采样 -> samples/step_%06d.txt" % step)

            if step % args.ckpt_every == 0:
                save_ckpt(last_path, model, opt, step, epoch, best_val, cfg, args)
                if args.keep_ckpt > 0:
                    arch = os.path.join(run_dir, "ckpt_%06d.pt" % step)
                    save_ckpt(arch, model, None, step, epoch, best_val, cfg, args, keep_opt=False)
                    olds = sorted([f for f in os.listdir(run_dir) if f.startswith("ckpt_")])
                    for f in olds[:-args.keep_ckpt]:
                        try:
                            os.remove(os.path.join(run_dir, f))
                        except Exception:
                            pass
                ctx.log("  存档 step=%d（last.pt）" % step)

    save_ckpt(last_path, model, opt, step, epoch, best_val, cfg, args)
    final = os.path.join(run_dir, "final.pt")
    save_ckpt(final, model, None, step, step // steps_per_epoch, best_val, cfg, args, keep_opt=False)
    ctx.log("结束：step=%d ｜ best_val=%s ｜ 用时 %.1f min ｜ 峰值显存 %.0f MB ｜ final.pt 已写"
            % (step, best_val, (time.time() - t_start) / 60,
               torch.cuda.max_memory_allocated() / 1048576 if torch.cuda.is_available() else 0.0))
    ctx.heartbeat({"pid": os.getpid(), "run": args.run, "step": step, "total_steps": total_steps,
                   "finished": True, "best_val": None if best_val == float("inf") else round(best_val, 4),
                   "tokens_per_step": tokens_per_step, "grad_ckpt": args.grad_ckpt, "updated": now()})


if __name__ == "__main__":
    main()
