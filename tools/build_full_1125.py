# -*- coding: utf-8 -*-
"""build_full_1125.py —— 只读提取 1123（含最新 C07/C09/C10）+ 1101 全量 1:1 混合语料，
并直接构建 token 缓存与初始打包数据（vocab_v2.json, max_len=1088）。

严格只读源目录（绝不写入/移动/修改）：
  1123: D:\\Ai_data\\1123_全api产出\\data_此文件夹内只能是存放预料产出！！！
  1101: D:\\Ai_data\\1101_新主题_全api产出\\产出（此文件内只能有干净的知识语料，其他文件不能放进去）

输出目录（全在 D:\\AI_xunlian\\1125_jingjichang\\dataset 下）：
  full_1125_raw\\     归一化去重后的 jsonl 分片 + _stats.json
  full_1125_cache\\   token 流缓存（供 train_arena.py 每轮自动重排）
  full_1125_packed\\  seed=1337 初始打包（train.bin / val.bin / meta.json）
"""
import collections
import hashlib
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from data import build_cache, repack  # noqa: E402

SRCS = [
    ("1123", r"D:\Ai_data\1123_全api产出\data_此文件夹内只能是存放预料产出！！！"),
    ("1101", r"D:\Ai_data\1101_新主题_全api产出\产出（此文件内只能有干净的知识语料，其他文件不能放进去）"),
]
RAW_OUT = os.path.join(ROOT, "dataset", "full_1125_raw")
CACHE_OUT = os.path.join(ROOT, "dataset", "full_1125_cache")
PACKED_OUT = os.path.join(ROOT, "dataset", "full_1125_packed")
VOCAB_PATH = os.path.join(ROOT, "za", "vocab_v2.json")

MAX_LEN = 1088
SEED = 1337
VAL_MOD = 200          # 0.5% 进 val
MIN_ANS_CHARS = 4
MIN_TOTAL_CHARS = 8

SKIP_SUB = ("_work", "batch_packs", "data_raw", "data_clean", "_archive_",
            "batch_outputs", "batch_rejected")
SKIP_NAME = ("_dup", "失败", "rejected", "_over_", "_samples", "topics_",
             "_tmp", "index", "索引")

ROLE_USER = {"user", "human", "提问", "问", "u"}
ROLE_ASST = {"assistant", "gpt", "bot", "回答", "答", "a"}


def cname(s):
    s = re.sub(r"[（(].*?[）)]", "", s)
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", s)
    s = s.strip("_")
    return s[:36] or "x"


def group_of(rel):
    parts = rel.split(os.sep)
    if len(parts) == 1:
        return "root"
    return cname(parts[0])


def pick_role(x):
    r = x.get("role")
    if r is None:
        r = x.get("from")
    r = str(r or "").strip().lower()
    if r in ROLE_USER:
        return "user"
    if r in ROLE_ASST:
        return "assistant"
    return None


def pick_text(x):
    for k in ("text", "content", "value"):
        v = x.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            buf = []
            for it in v:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    buf.append(it["text"])
                elif isinstance(it, str):
                    buf.append(it)
            if buf:
                return "".join(buf)
    return None


def norm_turns(seq):
    out = []
    for x in seq:
        if not isinstance(x, dict):
            continue
        role = pick_role(x)
        if role is None:
            continue
        txt = pick_text(x)
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if not txt:
            continue
        out.append({"role": role, "text": txt})
    return out


def normalize(row):
    ctx = row.get("context")
    if not isinstance(ctx, str) or not ctx.strip():
        ctx = None

    turns = None
    for k in ("turns", "messages", "conversations"):
        v = row.get(k)
        if isinstance(v, list) and v:
            turns = norm_turns(v)
            if turns:
                break

    text = row.get("text")
    text = text.strip() if isinstance(text, str) and text.strip() else None

    if turns and any(t["role"] == "assistant" for t in turns):
        if ctx is None and text:
            ctx = text
        return ("turns", turns, ctx)

    if text:
        return ("text", text, None)

    q = row.get("q") if isinstance(row.get("q"), str) else row.get("question")
    a = row.get("a") if isinstance(row.get("a"), str) else row.get("answer")
    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
        return ("turns", [{"role": "user", "text": q.strip()},
                          {"role": "assistant", "text": a.strip()}], ctx)

    return (None, None, None)


def row_ok(kind, payload):
    if kind == "text":
        return len(payload) >= MIN_TOTAL_CHARS
    ans = sum(len(t["text"]) for t in payload if t["role"] == "assistant")
    tot = sum(len(t["text"]) for t in payload)
    return ans >= MIN_ANS_CHARS and tot >= MIN_TOTAL_CHARS


def dedup_key(kind, payload, ctx):
    h = hashlib.md5()
    if ctx:
        h.update(ctx.encode("utf-8"))
        h.update(b"\x00")
    if kind == "text":
        h.update(payload.encode("utf-8"))
    else:
        for t in payload:
            h.update(t["role"].encode())
            h.update(b"\x01")
            h.update(t["text"].encode("utf-8"))
            h.update(b"\x02")
    return h.digest()[:8]


def extract_all():
    os.makedirs(RAW_OUT, exist_ok=True)
    # 清理旧分片避免残留
    for fn in os.listdir(RAW_OUT):
        if fn.endswith(".jsonl"):
            os.remove(os.path.join(RAW_OUT, fn))

    t0 = time.time()
    seen = set()
    writers = {}
    st = collections.defaultdict(lambda: collections.Counter())
    drop = collections.Counter()

    def writer_for(name):
        if name not in writers:
            p = os.path.join(RAW_OUT, name + ".jsonl")
            writers[name] = io.open(p, "w", encoding="utf-8")
        return writers[name]

    for src, base in SRCS:
        if not os.path.isdir(base):
            raise SystemExit("源不存在：%s" % base)
        files = []
        for root, dirs, fns in os.walk(base):
            rel_root = os.path.relpath(root, base)
            if any(s in rel_root for s in SKIP_SUB):
                continue
            for fn in fns:
                if not fn.endswith(".jsonl"):
                    continue
                if any(s in fn for s in SKIP_NAME):
                    continue
                files.append(os.path.join(root, fn))
        files.sort()
        print("[提取 %s] %d 个 jsonl 文件" % (src, len(files)), flush=True)
        for fi, fp in enumerate(files):
            rel = os.path.relpath(fp, base)
            grp = group_of(rel)
            name = "%s__%s" % (src, grp)
            fw = writer_for(name)
            nline = 0
            with io.open(fp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        drop["json_err"] += 1
                        continue
                    if not isinstance(row, dict):
                        drop["not_dict"] += 1
                        continue
                    nline += 1
                    kind, payload, ctx = normalize(row)
                    if kind is None:
                        drop["no_content"] += 1
                        continue
                    if not row_ok(kind, payload):
                        drop["too_short"] += 1
                        continue
                    k = dedup_key(kind, payload, ctx)
                    if k in seen:
                        drop["dup"] += 1
                        continue
                    seen.add(k)

                    sid = "%s-%s-%06d" % (src, grp, nline)
                    if (int.from_bytes(k, "big") % VAL_MOD) == 0:
                        sid = "val-" + sid
                    if kind == "text":
                        out = {"id": sid, "type": "text", "text": payload}
                        nch = len(payload)
                    else:
                        out = {"id": sid, "turns": payload}
                        if ctx:
                            out["context"] = ctx
                        nch = sum(len(t["text"]) for t in payload)
                    fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                    st[name]["rows"] += 1
                    st[name]["chars"] += nch
                    if sid.startswith("val-"):
                        st[name]["val_rows"] += 1
            if (fi + 1) % 60 == 0 or fi + 1 == len(files):
                print("  [%s %d/%d] %.1f min" % (src, fi + 1, len(files), (time.time() - t0) / 60), flush=True)

    for fw in writers.values():
        fw.close()

    tot_rows = sum(c["rows"] for c in st.values())
    tot_chars = sum(c["chars"] for c in st.values())
    tot_val = sum(c["val_rows"] for c in st.values())
    c1123 = sum(c["chars"] for k, c in st.items() if k.startswith("1123__"))
    c1101 = sum(c["chars"] for k, c in st.items() if k.startswith("1101__"))
    print("\n=== 提取完成：rows=%d | %.2f M字 (1123=%.2fM %.1f%%, 1101=%.2fM %.1f%%) | val=%d | drop=%s | %.1f min ==="
          % (tot_rows, tot_chars / 1e6, c1123 / 1e6, 100 * c1123 / max(1, tot_chars),
             c1101 / 1e6, 100 * c1101 / max(1, tot_chars), tot_val, dict(drop), (time.time() - t0) / 60),
          flush=True)

    stats = {"rows": tot_rows, "chars": tot_chars, "val_rows": tot_val,
             "chars_1123": c1123, "chars_1101": c1101,
             "drop": dict(drop), "shards": {k: dict(v) for k, v in st.items()},
             "srcs": [s[1] for s in SRCS], "elapsed_min": round((time.time() - t0) / 60, 2)}
    with io.open(os.path.join(RAW_OUT, "_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== Step 1/3: 只读提取 1123 + 1101 全量 1:1 语料 ===", flush=True)
    extract_all()

    print("\n=== Step 2/3: 分词并构建 token 缓存 (vocab_v2.json) ===", flush=True)
    build_cache(RAW_OUT, CACHE_OUT, max_len=MAX_LEN, vocab_path=VOCAB_PATH)

    print("\n=== Step 3/3: 初始打散打包 (seed=%d, max_len=%d) ===" % (SEED, MAX_LEN), flush=True)
    repack(CACHE_OUT, PACKED_OUT, seed=SEED, max_len=MAX_LEN)
    print("\nALL DONE!", flush=True)


if __name__ == "__main__":
    main()
