# -*- coding: utf-8 -*-
"""build_arena_50m.py —— 构建「1125 竞技场」~50M token 高质量偏向型测试集（方案 2）

来源（只读）：D:\AI_xunlian\1124\dataset\_mix\raw\*.jsonl
配比策略（目标 train ≈ 5,000 万 token，内部行级随机打散，val 100% 完整保留）：
  · 1123 新语料大类（C01~C08）：分层抽 35%（重测推理、属性五层、带资料定位、边界拒答）
  · 1101 基础知识型问答（老百科巨无霸）：压到 6%
  · 1101 其余大类（定位/概括/长资料/沅沅对话/常识）：分层抽 12%
  · 所有 < 5,000 行的极小高价值类（自我介绍/寒暄/C09/历史/物化等）：100% 保留
  · 所有 val- 样本：100% 保留（与 1124/packed_v2 完全同一把验证尺子）
"""
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from data import build_cache, repack_aligned, is_val  # noqa: E402
from tokenizer import default_vocab_path             # noqa: E402

RAW_DIR = r"D:\AI_xunlian\1124\dataset\_mix\raw"
OUT_RAW = os.path.join(ROOT, "dataset", "arena_50m_raw.jsonl")
OUT_CACHE = os.path.join(ROOT, "dataset", "arena_50m_cache")
OUT_PACK = os.path.join(ROOT, "dataset", "arena_50m")
SEED = 20261125
MAX_LEN = 1088


def keep_ratio(shard_name, n_train):
    if n_train < 5000:
        return 1.0
    if shard_name.startswith("1123__"):
        return 0.35
    if "基础知识型问答" in shard_name:
        return 0.06
    return 0.12


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    rng = random.Random(SEED)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl"))
    if not files:
        raise SystemExit("找不到源分片：%s" % RAW_DIR)

    train_lines = []
    val_lines = []
    summary = []
    for fn in files:
        shard = fn[:-6]
        p = os.path.join(RAW_DIR, fn)
        tr_buf, va_buf = [], []
        with io.open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if is_val(obj.get("id", "")):
                    va_buf.append(s)
                else:
                    tr_buf.append(s)
        ratio = keep_ratio(shard, len(tr_buf))
        if ratio < 1.0:
            k = max(1, int(round(len(tr_buf) * ratio)))
            picked = rng.sample(tr_buf, k)
        else:
            picked = tr_buf
        train_lines.extend(picked)
        val_lines.extend(va_buf)
        summary.append((shard, len(tr_buf), len(picked), ratio, len(va_buf)))

    # 内部全局随机打散 train 行
    rng.shuffle(train_lines)
    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
    with io.open(OUT_RAW, "w", encoding="utf-8") as f:
        for s in train_lines:
            f.write(s + "\n")
        for s in val_lines:
            f.write(s + "\n")

    print("=== 抽样完成（%.1fs）：train=%d 行（已随机打散），val=%d 行（100%% 保留）==="
          % (time.time() - t0, len(train_lines), len(val_lines)))
    for shard, tot, pk, r, nv in summary:
        print("  %-36s  %7d -> %6d (%4.0f%%)  val=%d" % (shard[:36], tot, pk, r * 100, nv))

    print("\n>>> 分词缓存 -> %s" % OUT_CACHE)
    build_cache(OUT_RAW, OUT_CACHE, max_len=MAX_LEN, vocab_path=default_vocab_path())
    print("\n>>> 行对齐打包 -> %s" % OUT_PACK)
    repack_aligned(OUT_CACHE, OUT_PACK, seed=SEED, max_len=MAX_LEN)
    print("\n=== 全部完成，总耗时 %.1fs ===" % (time.time() - t0))


if __name__ == "__main__":
    main()
