# -*- coding: utf-8 -*-
"""data.py —— 语料(jsonl) -> 定长 token 序列（含 loss 掩码）

序列模板（与 ESP32 端一致）：
    单轮  <bos><user>问<sep><assistant>答<eos>
    多轮  逐轮 <user>|<assistant> + 文本 + <sep>，尾部 <sep> 去掉再补 <eos>
    text 行 -> 一条 assistant 轮：<bos><assistant>陈述句<eos>
    RAG 行 -> context（资料，不参与 loss）+ 多轮问答

loss 掩码（SFT 口径）：用户段与结构标记 = -100；助手回答 / 条末 <sep> / <eos> 参与 loss。

打包：所有样本首尾相连成一个 token 流，再按 max_len 切窗（无 padding）。
    文档边界由 <bos>/<eos> 表达；跨文档注意力不做 mask（小模型预训练惯例）。

切分：id 以 `val-` 开头 → val（1125 主线做法）；否则按 id 的 md5 取模，默认 0.5% 进 val
     （`--val-ratio 0` 可关掉，固定不随运行变化）。

用法：
    python tools/data.py --src <语料目录或jsonl> --out dataset\\1029 --max-len 512
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tokenizer import Tokenizer, default_vocab_path  # noqa: E402

IGNORE = -100


def walk_jsonl(path):
    """支持：单个 .jsonl / 目录（按文件名排序）/ 有序列表（保持给定顺序）。"""
    if isinstance(path, (list, tuple)):
        out = []
        for p in path:
            out.extend(walk_jsonl(p))
        return out
    if os.path.isfile(path):
        return [path]
    out = []
    for base, _, fns in os.walk(path):
        for fn in fns:
            if fn.endswith(".jsonl"):
                out.append(os.path.join(base, fn))
    return sorted(out)


def segments(row):
    """-> [(文本, 是否算 loss)]，顺序与 build 一致。"""
    out = []
    if row.get("type") == "text" and isinstance(row.get("text"), str):
        out.append((row["text"], True))
        return out
    if isinstance(row.get("context"), str):
        out.append((row["context"], False))
    for t in (row.get("turns") or []):
        role, txt = t.get("role"), t.get("text", "")
        if not isinstance(txt, str):
            continue
        out.append((txt, role == "assistant"))
    return out


def is_val(sid, val_ratio=0.0):
    """val 判定。

    1125 主线的语料是显式留出：id 以 `val-` 开头（约 0.5%）。这里优先认这个前缀；
    外部把自己的 jsonl 直接丢进来时没有前缀，于是再按 id 的 md5 取模切一份 ——
    同一份数据、同一个比例，切出来的 val 永远一致（不依赖 Python 的 hash 随机种子）。
    """
    sid = str(sid)
    if sid.startswith("val-"):
        return True
    if val_ratio <= 0:
        return False
    h = int(hashlib.md5(sid.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000000) < int(val_ratio * 1000000)


def encode_row(row, tk):
    """一行 jsonl -> (ids, lbls)。

    模板：<bos>[<user>资料<sep>][<user>问<sep>][<assistant>答<sep> …]<eos>
      · <sep> = 轮次结束（“这句说完了”），每一段后面都有，位置固定，不随轮次变化
      · <eos> = 整条样本结束，只在行末出现
    助手段后的 <sep> 参与 loss（模型要学会“答完就收口”），用户段后的 <sep> 由程序补，不计 loss。
    """
    bos, eos, sep = tk.bos_id, tk.eos_id, tk.sep_id
    u_tok, a_tok = tk.user_id, tk.assistant_id
    ids, lbls = [bos], [IGNORE]
    for txt, ls in segments(row):
        ids.append(a_tok if ls else u_tok); lbls.append(IGNORE)
        enc = tk.encode(txt)
        ids.extend(enc)
        # 非 loss 段必须填 IGNORE(-100)；填 0 会被当成 <pad> 的真实目标。
        lbls.extend([IGNORE] * len(enc))
        if ls:
            start = len(lbls) - len(enc)
            for j in range(start, len(lbls)):
                lbls[j] = ids[j]
        ids.append(sep); lbls.append(sep if ls else IGNORE)
    ids.append(eos); lbls.append(eos)
    return ids, lbls


def build(src, out_dir, max_len=512, limit=0, quiet=False, vocab_path=None,
          val_ratio=0.005):
    """流式打包：token 边生成边写临时文件，最后按 max_len 切窗。

    每段话（资料/提问/回答）说完统一出 <sep>，行末补 <eos>；<sep> 是唯一的“说完了”信号。

    之前用 Python list 累积全部 token，语料超过 ~5M token 就会吃光内存；
    现在每满 FLUSH 个 token 就落一次盘（45M token 常驻内存 < 300MB）。
    """
    tk = Tokenizer(vocab_path or default_vocab_path())
    files = walk_jsonl(src)
    if not files:
        raise SystemExit("没有找到 .jsonl：%s" % src)
    n = {"train": 0, "val": 0}
    unk_total = 0
    FLUSH = 1 << 16

    os.makedirs(out_dir, exist_ok=True)
    tmp = {s: (os.path.join(out_dir, "_tmp_%s.bin" % s),
               os.path.join(out_dir, "_tmp_%s_lbl.bin" % s)) for s in ("train", "val")}
    fh = {s: (open(tmp[s][0], "wb"), open(tmp[s][1], "wb")) for s in ("train", "val")}
    pend = {"train": [[], []], "val": [[], []]}
    t0 = time.time()

    def flush(split):
        nonlocal unk_total
        ids, lbls = pend[split]
        if not ids:
            return
        a = np.asarray(ids, dtype=np.uint16)
        b = np.asarray(lbls, dtype=np.int16)
        unk_total += int(np.sum(a == tk.unk_id))
        fh[split][0].write(a.tobytes())
        fh[split][1].write(b.tobytes())
        ids.clear()
        lbls.clear()

    for fi, path in enumerate(files):
        with io.open(path, encoding="utf-8") as f:
            for line in f:
                if limit and (n["train"] + n["val"]) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not segments(row):
                    continue
                split = "val" if is_val(row.get("id", ""), val_ratio) else "train"
                r_ids, r_lbls = encode_row(row, tk)
                ids, lbls = pend[split]
                ids.extend(r_ids)
                lbls.extend(r_lbls)
                n[split] += 1
                if len(ids) >= FLUSH:
                    flush(split)
        flush("train"); flush("val")
        if not quiet:
            print("  [%d/%d] %s  train=%d val=%d ｜ %.1f min"
                  % (fi + 1, len(files), os.path.basename(path),
                     n["train"], n["val"], (time.time() - t0) / 60), flush=True)
    for s in ("train", "val"):
        fh[s][0].close(); fh[s][1].close()

    meta = {"max_len": max_len, "samples": n,
            "src": src if isinstance(src, str) else [str(p) for p in src],
            "vocab_size": tk.size, "vocab_path": str(vocab_path or default_vocab_path())}
    for split in ("train", "val"):
        p_ids, p_lbl = tmp[split]
        if not os.path.isfile(p_ids) or os.path.getsize(p_ids) == 0:
            # 空 split 也落一对空文件：PackedDataset 读到 0 窗口，训练端会跳过验证
            open(os.path.join(out_dir, "%s.bin" % split), "wb").close()
            open(os.path.join(out_dir, "%s_lbl.bin" % split), "wb").close()
            meta[split] = {"tokens": 0, "windows": 0, "loss_tokens": 0}
            print("  %s: 0 窗口（该 split 没有样本：检查 id 前缀或调 --val-ratio）" % split)
            continue
        ids = np.fromfile(p_ids, dtype=np.uint16)
        lbls = np.fromfile(p_lbl, dtype=np.int16)
        nw = len(ids) // max_len
        ids = ids[:nw * max_len].reshape(nw, max_len)
        lbls = lbls[:nw * max_len].reshape(nw, max_len)
        ids.tofile(os.path.join(out_dir, "%s.bin" % split))
        lbls.tofile(os.path.join(out_dir, "%s_lbl.bin" % split))
        meta[split] = {"tokens": int(nw * max_len), "windows": int(nw),
                       "loss_tokens": int(np.sum(lbls != IGNORE))}
        print("  %s: %d 窗口 x %d = %d token（loss %d）"
              % (split, nw, max_len, nw * max_len, meta[split]["loss_tokens"]))
        del ids, lbls
        os.remove(p_ids); os.remove(p_lbl)
    meta["unk_tokens"] = unk_total
    with io.open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("  unk token 总数：%d（>0 说明词表缺口）" % unk_total)
    print("  meta -> %s ｜ 用时 %.1f min" % (os.path.join(out_dir, "meta.json"),
                                          (time.time() - t0) / 60))
    return meta


def build_cache(src, out_dir, max_len=512, vocab_path=None, quiet=False):
    """分词一次，按「行」保存成 token 流缓存（供每轮重排后重新打包）。

    模板口径与 build() 一致：每段结束 <sep>，行末 <eos>。

    产出（out_dir 下）：
        cache_ids.bin   uint16  所有行的 token 首尾相连
        cache_lbl.bin   int16   对应标签（-100 / token id）
        cache_rows.npz  offsets(int64) / lengths(int32) / val(uint8)
    每行的排列顺序 == 源文件行顺序；重排交给 repack()，不必再分词。
    """
    tk = Tokenizer(vocab_path or default_vocab_path())
    files = walk_jsonl(src)
    if not files:
        raise SystemExit("没有找到 .jsonl：%s" % src)
    unk_total = 0
    FLUSH = 1 << 16
    os.makedirs(out_dir, exist_ok=True)
    fid = open(os.path.join(out_dir, "_tmp_ids.bin"), "wb")
    flb = open(os.path.join(out_dir, "_tmp_lbl.bin"), "wb")
    pend = [[], []]
    offsets, lengths, vals = [], [], []
    cur = 0
    t0 = time.time()

    def flush():
        nonlocal unk_total, cur
        if not pend[0]:
            return
        a = np.asarray(pend[0], dtype=np.uint16)
        b = np.asarray(pend[1], dtype=np.int16)
        unk_total += int(np.sum(a == tk.unk_id))
        fid.write(a.tobytes()); flb.write(b.tobytes())
        cur += len(a)
        pend[0].clear(); pend[1].clear()

    nrow = 0
    for fi, path in enumerate(files):
        with io.open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                segs = segments(row)
                if not segs:
                    continue
                ids, lbls = pend
                start_local = len(ids)          # 本行在未落盘缓冲里的起点
                r_ids, r_lbls = encode_row(row, tk)
                ids.extend(r_ids); lbls.extend(r_lbls)
                offsets.append(cur + start_local)   # 全局起点
                lengths.append(len(r_ids))
                vals.append(1 if is_val(row.get("id", "")) else 0)
                nrow += 1
                if len(ids) >= FLUSH:
                    flush()
        flush()
        if not quiet:
            print("  [%d/%d] %s  rows=%d  tokens=%d ｜ %.1f min"
                  % (fi + 1, len(files), os.path.basename(path), nrow, cur,
                     (time.time() - t0) / 60), flush=True)
    fid.close(); flb.close()
    with open(os.path.join(out_dir, "cache_rows.npz"), "wb") as f:
        np.savez(f, offsets=np.asarray(offsets, dtype=np.int64),
                 lengths=np.asarray(lengths, dtype=np.int32),
                 val=np.asarray(vals, dtype=np.uint8))
    os.replace(os.path.join(out_dir, "_tmp_ids.bin"), os.path.join(out_dir, "cache_ids.bin"))
    os.replace(os.path.join(out_dir, "_tmp_lbl.bin"), os.path.join(out_dir, "cache_lbl.bin"))
    meta = {"rows": nrow, "tokens": cur, "val_rows": int(sum(vals)),
            "unk_tokens": unk_total, "max_len": max_len, "src": files}
    with io.open(os.path.join(out_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("  缓存完成：%d 行 / %d token / val %d 行 / unk %d ｜ %.1f min"
          % (nrow, cur, meta["val_rows"], unk_total, (time.time() - t0) / 60))
    return meta


def repack(cache_dir, out_dir, seed, max_len=512, quiet=False):
    """按 seed 重新排列「行」，再切成 max_len 定长窗口写 train.bin / val.bin。

    同一份分词缓存 + 不同 seed = 每个 epoch 数据顺序都不同（不必重新分词）。
    val 行保持缓存顺序（不随 seed 变化），保证各轮 val_loss 可比。
    """
    rows = np.load(os.path.join(cache_dir, "cache_rows.npz"))
    offsets, lengths, vals = rows["offsets"], rows["lengths"], rows["val"]
    ids = np.memmap(os.path.join(cache_dir, "cache_ids.bin"), dtype=np.uint16, mode="r")
    lbls = np.memmap(os.path.join(cache_dir, "cache_lbl.bin"), dtype=np.int16, mode="r")
    if len(offsets) == 0:
        raise SystemExit("缓存为空：%s" % cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    meta = {"max_len": max_len, "seed": seed}
    for split in ("train", "val"):
        sel = np.where(vals == (1 if split == "val" else 0))[0]
        if split == "train":
            sel = sel[np.random.RandomState(seed).permutation(len(sel))]
        parts_i, parts_l = [], []
        for r in sel:
            o, l = int(offsets[r]), int(lengths[r])
            parts_i.append(np.asarray(ids[o:o + l], dtype=np.uint16))
            parts_l.append(np.asarray(lbls[o:o + l], dtype=np.int16))
        cat_i = np.concatenate(parts_i) if parts_i else np.zeros(0, dtype=np.uint16)
        cat_l = np.concatenate(parts_l) if parts_l else np.zeros(0, dtype=np.int16)
        nw = len(cat_i) // max_len
        cat_i = cat_i[:nw * max_len].reshape(nw, max_len)
        cat_l = cat_l[:nw * max_len].reshape(nw, max_len)
        cat_i.tofile(os.path.join(out_dir, "%s.bin" % split))
        cat_l.tofile(os.path.join(out_dir, "%s_lbl.bin" % split))
        meta[split] = {"rows": int(len(sel)), "windows": int(nw),
                       "tokens": int(nw * max_len),
                       "loss_tokens": int(np.sum(cat_l != IGNORE))}
        if not quiet:
            print("  %s: %d 行 -> %d 窗口 x %d = %d token（loss %d）"
                  % (split, len(sel), nw, max_len, nw * max_len,
                     meta[split]["loss_tokens"]))
    with io.open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if not quiet:
        print("  重排打包 seed=%d ｜ %.1fs -> %s" % (seed, time.time() - t0, out_dir))
    return meta


def repack_aligned(cache_dir, out_dir, seed, max_len=1056, quiet=False):
    """行对齐装箱：整行绝不拦腰切断；短行拼满一窗，剩余位置补 <eos>（labels IGNORE）。

    与 repack() 的区别：旧流式会把长行切成两窗，导致后段答案在训练时看不到资料。
    这里每个答案 token 必然看得到自己那一行的资料与全部前轮问答。
    要求：所有行长度 <= max_len（1109 全语料最长 1048，1056 窗可全覆盖）。
    """
    rows = np.load(os.path.join(cache_dir, "cache_rows.npz"))
    offsets, lengths, vals = rows["offsets"], rows["lengths"], rows["val"]
    ids = np.memmap(os.path.join(cache_dir, "cache_ids.bin"), dtype=np.uint16, mode="r")
    lbls = np.memmap(os.path.join(cache_dir, "cache_lbl.bin"), dtype=np.int16, mode="r")
    if len(offsets) == 0:
        raise SystemExit("缓存为空：%s" % cache_dir)
    over = int(np.sum(lengths > max_len))
    if over:
        raise SystemExit("有 %d 行超过 max_len=%d，行对齐装箱不允许切行" % (over, max_len))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    meta = {"max_len": max_len, "seed": seed, "aligned": True}
    PAD = 3  # <eos>，labels 为 IGNORE，仅占位
    FLUSH_W = 512
    for split in ("train", "val"):
        sel = np.where(vals == (1 if split == "val" else 0))[0]
        if split == "train":
            sel = sel[np.random.RandomState(seed).permutation(len(sel))]
        fi = open(os.path.join(out_dir, "%s.bin" % split), "wb")
        fl = open(os.path.join(out_dir, "%s_lbl.bin" % split), "wb")
        buf_i = np.full(max_len, PAD, dtype=np.uint16)
        buf_l = np.full(max_len, IGNORE, dtype=np.int16)
        pos = 0
        nw = 0
        loss_tokens = 0
        pad_tokens = 0
        pend_i, pend_l = [], []

        def flush():
            if not pend_i:
                return
            np.asarray(pend_i, dtype=np.uint16).tofile(fi)
            np.asarray(pend_l, dtype=np.int16).tofile(fl)
            pend_i.clear(); pend_l.clear()

        for r in sel:
            o, l = int(offsets[r]), int(lengths[r])
            if pos + l > max_len:                      # 当前窗放不下 → 封窗
                pend_i.append(buf_i.reshape(1, max_len).copy())
                pend_l.append(buf_l.reshape(1, max_len).copy())
                nw += 1
                buf_i.fill(PAD); buf_l.fill(IGNORE)
                pos = 0
                if nw % FLUSH_W == 0:
                    flush()
            buf_i[pos:pos + l] = ids[o:o + l]
            buf_l[pos:pos + l] = lbls[o:o + l]
            pos += l
        if pos:                                        # 尾窗（不满也写，pad 占位）
            pend_i.append(buf_i.reshape(1, max_len).copy())
            pend_l.append(buf_l.reshape(1, max_len).copy())
            nw += 1
        flush()
        fi.close(); fl.close()
        all_lbl = np.memmap(os.path.join(out_dir, "%s_lbl.bin" % split),
                            dtype=np.int16, mode="r")
        loss_tokens = int(np.sum(all_lbl != IGNORE))
        pad_tokens = int(nw * max_len - sum(int(lengths[r]) for r in sel))
        meta[split] = {"rows": int(len(sel)), "windows": int(nw),
                       "tokens": int(nw * max_len), "loss_tokens": loss_tokens,
                       "pad_tokens": pad_tokens,
                       "pad_ratio": round(pad_tokens / max(nw * max_len, 1), 4)}
        if not quiet:
            print("  %s: %d 行 -> %d 窗口 x %d = %d token（loss %d，pad %.1f%%）"
                  % (split, len(sel), nw, max_len, nw * max_len,
                     loss_tokens, 100.0 * pad_tokens / max(nw * max_len, 1)))
    with io.open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if not quiet:
        print("  行对齐打包 seed=%d max_len=%d ｜ %.1fs -> %s"
              % (seed, max_len, time.time() - t0, out_dir))
    return meta


class PackedDataset(object):
    def __init__(self, out_dir, split="train", max_len=512):
        self.dir, self.split, self.max_len = out_dir, split, max_len
        self.reload()

    def reload(self):
        p_ids = os.path.join(self.dir, "%s.bin" % self.split)
        p_lbl = os.path.join(self.dir, "%s_lbl.bin" % self.split)
        if not (os.path.isfile(p_ids) and os.path.isfile(p_lbl)):
            self.ids = np.zeros(0, dtype=np.uint16)
            self.lbls = np.zeros(0, dtype=np.int16)
            self.n = 0
            return self
        self.ids = np.fromfile(p_ids, dtype=np.uint16)
        self.lbls = np.fromfile(p_lbl, dtype=np.int16)
        self.n = len(self.ids) // self.max_len
        self.ids = self.ids[:self.n * self.max_len].reshape(self.n, self.max_len)
        self.lbls = self.lbls[:self.n * self.max_len].reshape(self.n, self.max_len)
        return self

    def __len__(self):
        return self.n

    def get_batch(self, idx, device=None, dtype=None):
        import torch
        x = torch.from_numpy(self.ids[idx].astype(np.int64))
        y = torch.from_numpy(self.lbls[idx].astype(np.int64))
        if device is not None:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+",
                    help="语料路径：可给多个 jsonl（按给定顺序拼接），也可给目录（按文件名排序）")
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "1029"))
    ap.add_argument("--vocab", default=default_vocab_path())
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.005,
                    help="留出比例：id 不以 val- 开头时按 md5 取模切分（默认 0.5%%）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src = args.src[0] if len(args.src) == 1 else list(args.src)
    print("构建数据集：%s -> %s ｜ vocab=%s" % (src, args.out, args.vocab))
    build(src, args.out, max_len=args.max_len, limit=args.limit, vocab_path=args.vocab,
          val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
