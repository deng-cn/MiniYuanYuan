# -*- coding: utf-8 -*-
"""eval_ep2_deep.py —— 在 CPU 上对第 2 轮最佳模型 (19.71M, val=1.8439) 进行全面深度能力体检，
零占用 GPU 显存，不干扰第 3 轮 MoE 训练。

输出两部分：
  1) 全语料 20+ 个分片留出 val 样本的逐类 Cross-Entropy 雷达图（精确看哪里强哪里弱）
  2) 9 大实战维度共 42 道真实测试题（含多轮对话、长短RAG、拒答、语言指令、常识、逻辑、人设）的逐字回答。
"""
import glob
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "moxing"))

from data import encode_row, segments, IGNORE  # noqa: E402
from model_p4 import LMP4, cfg_p4               # noqa: E402
from tokenizer import Tokenizer                 # noqa: E402

CKPT = os.path.join(ROOT, "out", "m12l_d384_full1125_ep2", "best.pt")
RAW_DIR = os.path.join(ROOT, "dataset", "full_1125_raw")
VOCAB = os.path.join(ROOT, "za", "vocab_v2.json")
OUT_REPORT = os.path.join(ROOT, "out", "m12l_d384_full1125_ep2", "deep_eval_ep2.txt")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    torch.set_num_threads(max(4, min(12, os.cpu_count() or 8)))
    t0 = time.time()

    tk = Tokenizer(VOCAB)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = cfg_p4(**ck["cfg"])
    cfg["attn_backend"] = "auto"
    model = LMP4(cfg).float().eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})

    lines = []
    def p(s=""):
        lines.append(s)
        print(s, flush=True)

    p("=" * 88)
    p(f"第 2 轮终版模型深度能力体检 | ckpt={os.path.basename(CKPT)} | step={ck.get('step')} | val={ck.get('best_val'):.4f} | params={model.num_params():,}")
    p("=" * 88)

    # ---- Part 1: 逐分片留出 val Cross-Entropy ----
    p("\n【一、24 个语料分片留出验证集 Cross-Entropy 分布（每片抽最多 24 条 val-）】")
    shard_res = []
    with torch.no_grad():
        for sp in sorted(glob.glob(os.path.join(RAW_DIR, "*.jsonl"))):
            name = os.path.basename(sp).replace(".jsonl", "")
            tot_loss, tot_tok, n_rows = 0.0, 0, 0
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if not str(row.get("id", "")).startswith("val-"):
                        continue
                    ids, lbls = encode_row(row, tk)
                    if len(ids) > 1088:
                        ids, lbls = ids[:1088], lbls[:1088]
                    if sum(1 for x in lbls[1:] if x != IGNORE) == 0:
                        continue
                    x = torch.tensor([ids], dtype=torch.long)
                    y = torch.tensor([lbls], dtype=torch.long)
                    out = model(x, labels=y)
                    nt = int((y[:, 1:] != IGNORE).sum())
                    tot_loss += float(out["loss"]) * nt
                    tot_tok += nt
                    n_rows += 1
                    if n_rows >= 24:
                        break
            if tot_tok > 0:
                ce = tot_loss / tot_tok
                shard_res.append((ce, n_rows, tot_tok, name))

    shard_res.sort()
    for ce, nr, nt, name in shard_res:
        ppl = math.exp(min(20.0, ce))
        p(f"  CE={ce:6.3f} (PPL={ppl:6.1f}) | n={nr:2d} rows, {nt:5d} loss_tok | {name}")

    # ---- Part 2: 9 大实战维度真实肉眼评测 ----
    @torch.no_grad()
    def ask_turns(turns, ctx=None, max_new=96):
        text = "<bos>"
        if ctx:
            text += "<user>" + ctx + "<sep>"
        for role, msg in turns:
            tag = "<user>" if role == "user" else "<assistant>"
            text += tag + msg + "<sep>"
        text += "<assistant>"
        ids = torch.tensor([tk.encode(text)], dtype=torch.long)
        r = model.generate(
            ids, max_new_tokens=max_new, temperature=0.0, do_sample=False, top_k=1,
            eos_token_id=[tk.eos_id, tk.stoi["<sep>"]],
            repetition_penalty=1.08, no_repeat_ngram_size=4, context_len=ids.shape[1]
        )
        return tk.decode(r["new_tokens"], skip_special_tokens=True).strip()

    @torch.no_grad()
    def cont(prefix, max_new=64):
        text = "<bos><assistant>" + prefix
        ids = torch.tensor([tk.encode(text)], dtype=torch.long)
        r = model.generate(
            ids, max_new_tokens=max_new, temperature=0.0, do_sample=False, top_k=1,
            eos_token_id=[tk.eos_id, tk.stoi["<sep>"]],
            repetition_penalty=1.08, no_repeat_ngram_size=4, context_len=ids.shape[1]
        )
        return tk.decode(r["new_tokens"], skip_special_tokens=True).strip()

    p("\n" + "=" * 88)
    p("【二、9 大实战能力维度逐题实测（全部 temperature=0 贪心解码，肉眼直评）】")
    p("=" * 88)

    CASES = [
        # 1. 长文 RAG 精准定位
        ("1-RAG长文定位", "转运中心位于城南工业园B座，每晚十一点停止收件；生鲜冷链必须走二号月台，普通快递走四号月台。昨晚值班队长是周海涛，由于暴雨导致一号门积水，所有货车临时改从东侧三号门进出。",
         [("user", "生鲜冷链要走几号月台？")]),
        ("1-RAG长文定位", "转运中心位于城南工业园B座，每晚十一点停止收件；生鲜冷链必须走二号月台，普通快递走四号月台。昨晚值班队长是周海涛，由于暴雨导致一号门积水，所有货车临时改从东侧三号门进出。",
         [("user", "昨晚因为暴雨，货车改从哪个门进出？")]),
        ("1-RAG长文多轮", "社区图书馆周二闭馆整理，周三到周日每天上午九点开馆、晚上八点半闭馆。读者每次最多借五本书，借期二十一天；二楼东区是少儿绘本角，西区是安静自习室。",
         [("user", "图书馆哪天不开门？"), ("assistant", "周二闭馆整理。"), ("user", "那去二楼自习的话应该往东边走还是西边走？")]),
        ("1-RAG中篇C07", None,
         [("user", "请阅读下面这段资料：\n古时候走夜路没有路灯，人们常在竹篾扎成的灯笼外糊上一层涂过桐油的绵纸。桐油既能让纸面透亮，还能挡住夜里的湿气和细雨，点上蜡烛后提在手里，风吹不灭、雨打不湿，照亮脚下两三步的石板路。\n\n灯笼外面糊的纸为什么要涂桐油？")]),

        # 2. RAG 忠实拒答（资料未提及时不瞎编）
        ("2-RAG防瞎编", "本次体检一共设了三个项目：身高体重、视力和心电图。张阿姨早上八点半空腹去的医院，九点十分就全部查完了。",
         [("user", "张阿姨的血糖结果正常吗？")]),
        ("2-RAG防瞎编C07", None,
         [("user", "请阅读下面这段资料：\n老李家的小菜园种了黄瓜、茄子和西红柿，每天清晨他都会挑两担井水来浇菜，到了傍晚还要顺手捉一遍菜叶背面的青虫。\n\n老李每个月卖菜能挣多少钱？")]),

        # 3. RAG 多点概括总结
        ("3-RAG归纳总结", "老城区改造方案如下：第一阶段先修地下排水管网，避免夏天内涝；第二阶段给老旧楼梯加装扶手和夜间感应灯；第三阶段把中心广场的破损地砖全部换成防滑透水砖，并在东角增设六张纳凉长椅。整个工程预计明年三月动工，十月底前完工。",
         [("user", "请用一句话概括老城区改造主要做哪三件事，以及什么时候完工。")]),
        ("3-RAG归纳总结", "周六早上八点在东门集合去青龙山徒步，全程十二公里，预计下午四点返回。每人需自备两升水、防滑鞋和雨衣，山顶信号弱。",
         [("user", "去徒步前每个人要自己准备哪三样东西？")]),

        # 4. 真实人设与自我边界
        ("4-人设与边界", None, [("user", "你叫什么名字，是哪里来的？")]),
        ("4-人设与边界", None, [("user", "你能帮我点一份外卖送到公司吗？")]),
        ("4-人设与边界", None, [("user", "帮我查一下今天下午三点从长沙到北京的高铁还有没有票。")]),
        ("4-人设与边界", None, [("user", "你平时喜欢吃什么菜，爱吃辣吗？")]),

        # 5. 日常对话与情绪共情（重点测 C08 拒答模板是否误伤）
        ("5-情绪对话", None, [("user", "今天被领导当众批评了一顿，心里特别委屈。")]),
        ("5-情绪对话", None, [("user", "忙了一整天终于下班回家了，好开心啊！")]),
        ("5-日常闲聊多轮", None, [("user", "最近老是熬夜刷手机，第二天起来头疼。"), ("assistant", "熬夜刷手机最耗神了，第二天头疼也是身体在提醒你该早点休息。你一般是睡前躺着刷吗？"), ("user", "是啊，一刷就停不下来，有什么好办法能早点睡？")]),
        ("5-日常闲聊", None, [("user", "晚上吃啥好呢，给点家常菜灵感？")]),

        # 6. C10 语言任务（改写/抽取/分类/排序）
        ("6-C10改写", None, [("user", "把下面这句话改写得更温和礼貌一点：赶紧把桌子收拾干净，乱死了。")]),
        ("6-C10抽取", None, [("user", "请只提取下面句子里的时间、人物和地点：本周三下午两点，王老师在二楼阶梯教室开家长会。")]),
        ("6-C10归类", None, [("user", "把下面四样东西分成水果和蔬菜两类：黄瓜、香蕉、菠菜、草莓。")]),
        ("6-C10排序", None, [("user", "把春、冬、秋、夏按一年里的先后顺序排一下。")]),

        # 7. C04/C10 实物属性对比与描述
        ("7-实物比较", None, [("user", "大象和蚂蚁比，体型差距有多大？")]),
        ("7-实物比较", None, [("user", "羽绒服和短袖T恤一般分别在什么季节穿？")]),
        ("7-实物属性", None, [("user", "玻璃杯最怕什么，平时用的时候要注意啥？")]),
        ("7-实物比较", None, [("user", "西瓜和樱桃哪个更重、个头更大？")]),

        # 8. 百科常识（查串台和实体记忆强度）
        ("8-百科常识", None, [("user", "中国首都是哪座城市？")]),
        ("8-百科常识", None, [("user", "地球围绕太阳转还是太阳围绕地球转？")]),
        ("8-百科常识", None, [("user", "一年有几个季节，分别是哪几个？")]),
        ("8-百科常识", None, [("user", "人体正常体温大概是多少度？")]),
        ("8-百科常识", None, [("user", "《静夜思》是谁写的诗？")]),
        ("8-生活常识", None, [("user", "炒菜时油锅突然起火了，能直接泼冷水灭火吗？")]),

        # 9. 数字、算术与简单逻辑推理
        ("9-算术逻辑", None, [("user", "盒子里原来有8个苹果，拿走3个，还剩几个？")]),
        ("9-逻辑比较", None, [("user", "小明比小红高，小刚比小明高，这三个人里谁最高？")]),
        ("9-逻辑判断", None, [("user", "铁块和木块扔进水里，通常哪个会沉下去，哪个会浮起来？")]),
        ("9-因果逻辑", None, [("user", "如果屋里一直开着空调还把窗户大敞着，会有什么结果？")]),
    ]

    for cat, ctx, turns in CASES:
        ans = ask_turns(turns, ctx=ctx, max_new=96)
        q_show = turns[-1][1].replace("\n", " ")
        if len(turns) > 1:
            q_show = f"(多轮第{len(turns)//2+1}问) " + q_show
        p(f"\n[{cat}] Q: {q_show}")
        if ctx:
            p(f"  [资料] {ctx[:65]}{'...' if len(ctx)>65 else ''}")
        p(f"  [回答] {ans}")

    p("\n" + "=" * 88)
    p("【三、纯文本续写测试（故事叙事 & 景色描写）】")
    p("=" * 88)
    for pref in ["傍晚时分，橘红色的晚霞铺满了整个河面，", "老木匠拿起刨子，对着那块散发着淡淡松香的木料"]:
        p(f"\n[续写] 开头: {pref}")
        p(f"  [生成] {cont(pref, max_new=80)}")

    p(f"\n评测总耗时: {time.time() - t0:.1f}s")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    p(f"完整报告已保存 -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
