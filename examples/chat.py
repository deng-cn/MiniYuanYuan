# -*- coding: utf-8 -*-
"""最小可运行的对话示例。

用法：
    python examples/chat.py --prompt "你好，你是谁？"
    python examples/chat.py --model weights/m14l_d512_78m_ep2_fp16 --prompt "你好，你是谁？"
    python examples/chat.py --model weights/m12l_d384_ep2_fp16.pt --prompt "帮我改一下病句：我家客厅三十平方米。"

模型选择：
    weights/m14l_d512_78m_ep2_fp16      全量 14 层独立 BiCrossGate，78.58M（4 片 FP16，含 C13/SG）
    weights/m12l_d384_moe_ep3_fp16.pt   第 3 轮 MoE，30.34M
    weights/m12l_d384_ep2_fp16.pt       第 2 轮 12 层共享，19.71M
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "moxing"))

from tokenizer import Tokenizer  # noqa: E402


def load_checkpoint(model_path):
    """兼容单文件 .pt 与分片目录（含 meta.json + shard_*.pt）。"""
    if os.path.isdir(model_path) or model_path.endswith(".json"):
        meta_path = os.path.join(model_path, "meta.json") if os.path.isdir(model_path) else model_path
        base_dir = os.path.dirname(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            ck = json.load(f)
        sd = {}
        for sh in ck["shards"]:
            sd.update(torch.load(os.path.join(base_dir, sh), map_location="cpu", weights_only=True))
        ck["model"] = sd
        return ck
    return torch.load(model_path, map_location="cpu", weights_only=False)


def build(model_path):
    ck = load_checkpoint(model_path)
    cfg = ck["cfg"]
    if cfg.get("arch_type") == "std_gqa_bicross_moe":
        from model_moe import LMMoE, cfg_moe
        model = LMMoE(cfg_moe(**cfg))
    else:
        from model_p4 import LMP4, cfg_p4
        model = LMP4(cfg_p4(**cfg))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.inference_mode()
def chat(model, tk, query, context=None, temp=0.0, max_new=160, device="cpu"):
    prompt = "<bos>"
    if context:
        prompt += "<user>%s<sep>" % context
    prompt += "<user>%s<sep><assistant>" % query

    ids = torch.tensor([tk.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=max_new, temperature=temp,
        do_sample=(temp > 0), top_p=0.9, top_k=50 if temp > 0 else 1,
        eos_token_id=[tk.eos_id, tk.stoi["<sep>"]],
        repetition_penalty=1.08, no_repeat_ngram_size=4,
        context_len=ids.shape[1],
    )
    new = out["new_tokens"] if isinstance(out, dict) else out[0][ids.shape[1]:]
    return tk.decode(new, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "m12l_d384_moe_ep3_fp16.pt"))
    ap.add_argument("--vocab", default=os.path.join(ROOT, "corpus", "vocab_v2.json"))
    ap.add_argument("--prompt", default="你好，你是谁？")
    ap.add_argument("--context", default="", help="前置资料（模拟 RAG）")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = 贪心解码")
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tk = Tokenizer(args.vocab)
    model, ck = build(args.model)
    model.to(device).eval()

    print("模型：%s（step %s，best_val %.4f，dtype %s）"
          % (os.path.basename(args.model), ck.get("step"),
             ck.get("best_val") or 0, ck.get("dtype", "?")))
    print("设备：%s" % device)
    print("-" * 60)
    print(chat(model, tk, args.prompt, args.context or None,
               temp=args.temp, max_new=args.max_new, device=device))


if __name__ == "__main__":
    main()
