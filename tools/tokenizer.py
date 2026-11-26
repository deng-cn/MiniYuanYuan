# -*- coding: utf-8 -*-
"""tokenizer.py —— 确定性最长匹配分词器（与 ESP32 C 端同语义）

★ 不用 BPE：BPE 合并表不可能正好等于词表清单，中文上还可能把汉字切成 byte token，
  「每个汉字恰好 1 token」的前提就废了。最长匹配能逐条断言不变量。

三条不变量（tools/smoke.py 会真跑）：
    ① 词表内每个汉字 → 恰好 1 token
    ② 词表内每个 1-token 词 → 恰好 1 token
    ③ decode(encode(x)) == x（x 只含词表内字符时）

词表查找顺序（可用 --vocab 或环境变量 VOCAB_PATH 覆盖）：
    za/vocab_v2.json  →  za/vocab.json  →  corpus/vocab_v2.json  →  vocab.json（根）  →  out/vocab.json
"""
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def default_vocab_path():
    env = os.environ.get("VOCAB_PATH")
    if env and os.path.isfile(env):
        return env
    for p in (os.path.join(ROOT, "za", "vocab_v2.json"),
              os.path.join(ROOT, "za", "vocab.json"),
              os.path.join(ROOT, "corpus", "vocab_v2.json"),
              os.path.join(ROOT, "vocab.json"),
              os.path.join(ROOT, "out", "vocab.json")):
        if os.path.isfile(p):
            return p
    return os.path.join(ROOT, "za", "vocab_v2.json")


DEFAULT_VOCAB = default_vocab_path()


def load_vocab(path=None):
    with io.open(path or DEFAULT_VOCAB, encoding="utf-8") as f:
        return json.load(f)


def _encode_with_specials(text, sp_re, encode_span, stoi):
    if sp_re is None:
        return list(encode_span(text))
    ids, pos = [], 0
    for m in sp_re.finditer(text):
        if m.start() > pos:
            ids.extend(encode_span(text[pos:m.start()]))
        ids.append(stoi[m.group(0)])
        pos = m.end()
    if pos < len(text):
        ids.extend(encode_span(text[pos:]))
    return ids


class Tokenizer(object):
    def __init__(self, vocab=None, specials_as_text=True, map_whitespace=True):
        data = vocab if isinstance(vocab, dict) else load_vocab(vocab)
        self.tokens = list(data["tokens"])
        self.kinds = list(data.get("kinds") or [])
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.size = len(self.tokens)
        self.specials = dict(data["specials"])
        self.unk_id = self.stoi["<unk>"]
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.sep_id = self.stoi["<sep>"]
        self.user_id = self.stoi["<user>"]
        self.assistant_id = self.stoi["<assistant>"]
        self.max_len = max(len(t) for t in self.tokens if t not in self.specials)
        self.specials_as_text = specials_as_text
        if specials_as_text:
            pat = "|".join(re.escape(t) for t in sorted(self.specials, key=len, reverse=True))
            self._sp_re = re.compile(pat)
        else:
            self._sp_re = None
        self.map_whitespace = map_whitespace
        self._ws_map = ({" ": self.stoi["<space>"], "\n": self.stoi["<newline>"],
                         "\t": self.stoi["<space>"]} if map_whitespace else None)

    # ---------------- 编码 ----------------
    def _encode_span(self, s, unknown=None):
        ids, i, n = [], 0, len(s)
        maxlen, stoi, ws = self.max_len, self.stoi, self._ws_map
        while i < n:
            if ws is not None:
                j0 = ws.get(s[i])
                if j0 is not None:
                    ids.append(j0); i += 1; continue
            for L in range(min(maxlen, n - i), 0, -1):
                j = stoi.get(s[i:i + L])
                if j is not None:
                    ids.append(j); i += L; break
            else:
                ids.append(self.unk_id)
                if unknown is not None:
                    unknown.append(s[i])
                i += 1
        return ids

    def encode(self, text, bos=False, eos=False, unknown=None):
        ids = _encode_with_specials(text, self._sp_re,
                                    lambda chunk: self._encode_span(chunk, unknown),
                                    self.stoi)
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def encode_with_report(self, text):
        bad = []
        ids = self.encode(text, unknown=bad)
        return ids, bad

    # ---------------- 解码 ----------------
    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            t = self.tokens[i]
            if skip_special_tokens and t in self.specials:
                continue
            out.append(t)
        return "".join(out)

    def decode_tokens(self, ids):
        return [self.tokens[i] for i in ids]

    def __call__(self, text):
        return self.encode(text)

    def __len__(self):
        return self.size


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tk = Tokenizer()
    print("词表 %s ｜ %d 项" % (DEFAULT_VOCAB, len(tk)))
    for s in ("苹果是什么？", "你好，你是谁？"):
        ids, bad = tk.encode_with_report(s)
        print("  %-12s -> %d token ｜ unk %s ｜ roundtrip %s"
              % (s, len(ids), bad, tk.decode(ids) == s))
