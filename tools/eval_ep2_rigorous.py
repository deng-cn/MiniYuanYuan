# -*- coding: utf-8 -*-
"""eval_ep2_rigorous.py —— 三层对照严格科学评测（原题记忆 -> 换问法理解 -> 经全库检索验证的0命中新组合）

零 GPU 显存占用（纯 CPU 跑推理），不干扰第 3 轮 MoE GPU 训练。
注意：按唯一文本指纹精确锁定 Tier-A 原题（避免同分片多子文件 id 重号）。
"""
import glob
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "moxing"))

from model_p4 import LMP4, cfg_p4  # noqa: E402
from tokenizer import Tokenizer    # noqa: E402

CKPT = os.path.join(ROOT, "out", "m12l_d384_full1125_ep2", "best.pt")
RAW_DIR = os.path.join(ROOT, "dataset", "full_1125_raw")
VOCAB = os.path.join(ROOT, "za", "vocab_v2.json")
OUT_TXT = os.path.join(ROOT, "out", "m12l_d384_full1125_ep2", "rigorous_eval_ep2.txt")

# 用唯一文本片段锁定训练集原题（保证与 Tier-B 换问法 100% 为同一道原题）
ROW_MATCHERS = {
    "C01": lambda r: not r["id"].startswith("val-") and "人物用陈伯" in r.get("turns", [{}])[0].get("text", ""),
    "C02": lambda r: not r["id"].startswith("val-") and "说屋子复杂还是繁杂" in r.get("turns", [{}])[0].get("text", ""),
    "C03": lambda r: not r["id"].startswith("val-") and "买的这条皮带，是不是跟萝卜一样" in r.get("turns", [{}])[0].get("text", ""),
    "C04": lambda r: not r["id"].startswith("val-") and "吃四季豆是要去皮的吧" in r.get("turns", [{}])[0].get("text", ""),
    "C05": lambda r: not r["id"].startswith("val-") and "中国教师节是哪天来着" in r.get("turns", [{}])[0].get("text", ""),
    "C06": lambda r: not r["id"].startswith("val-") and "林小满、沈青禾、慧姨" in r.get("turns", [{}])[0].get("text", ""),
    "C07": lambda r: not r["id"].startswith("val-") and "银行营业厅作为公共场所" in r.get("turns", [{}])[0].get("text", ""),
    "C08": lambda r: not r["id"].startswith("val-") and "肯定记得上周咱俩聊过啥" in r.get("turns", [{}])[0].get("text", ""),
    "C09": lambda r: not r["id"].startswith("val-") and "菜场的青菜现在什么价" in r.get("turns", [{}])[0].get("text", ""),
    "C10A": lambda r: not r["id"].startswith("val-") and "杜小北把黄瓜放到了茶几上" in r.get("turns", [{}])[0].get("text", ""),
    "C10B": lambda r: not r["id"].startswith("val-") and "芝麻、苹果、篮球、书桌，这四个东西哪个最大" in r.get("turns", [{}])[0].get("text", ""),
    "C10C": lambda r: not r["id"].startswith("val-") and "无论天气多热，牛奶都得冷藏保存" in r.get("turns", [{}])[0].get("text", ""),
    "1101_LOC": lambda r: not r["id"].startswith("val-") and "虎皮青椒用的是什么辣椒" in r.get("turns", [{}])[0].get("text", ""),
    "1101_SUM": lambda r: not r["id"].startswith("val-") and "压面器这个东西主要是干什么用的" in r.get("turns", [{}])[0].get("text", ""),
    "1101_BASE": lambda r: not r["id"].startswith("val-") and "独头蒜和普通蒜放一起能看出来吗" in r.get("turns", [{}])[0].get("text", ""),
    "1101_YUAN": lambda r: not r["id"].startswith("val-") and "那套房子卖亏了" in r.get("turns", [{}])[0].get("text", ""),
}

ROW_KEYWORDS = {
    "C01": "人物用陈伯",
    "C02": "说屋子复杂还是繁杂",
    "C03": "买的这条皮带",
    "C04": "吃四季豆是要去皮",
    "C05": "中国教师节是哪天",
    "C06": "林小满、沈青禾、慧姨",
    "C07": "银行营业厅作为公共场所",
    "C08": "肯定记得上周咱俩聊过啥",
    "C09": "菜场的青菜现在什么价",
    "C10A": "杜小北把黄瓜放到",
    "C10B": "芝麻、苹果、篮球、书桌",
    "C10C": "无论天气多热，牛奶都得冷藏保存",
    "1101_LOC": "虎皮青椒用的是什么辣椒",
    "1101_SUM": "压面器这个东西主要",
    "1101_BASE": "独头蒜和普通蒜放一起",
    "1101_YUAN": "那套房子卖亏了",
}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    torch.set_num_threads(max(4, min(12, os.cpu_count() or 8)))
    t0 = time.time()

    fetched = {}
    corpus_checks = {
        "C01_新组合(陶艺+竹蜻蜓+琥珀)": lambda s: ("陶艺" in s and "竹蜻蜓" in s and "琥珀" in s),
        "C02_新组合(愤怒+手舞足蹈)": lambda s: ("愤怒" in s and "手舞足蹈" in s),
        "C03_新组合(登山袜+腐殖土)": lambda s: ("登山袜" in s and "腐殖土" in s),
        "C04_新组合(木陀螺+装水)": lambda s: ("木陀螺" in s and "装水" in s),
        "C05_新组合(星期五天+双休日不算)": lambda s: ("双休日" in s and "不算进星期" in s),
        "C06_新组合(赵阿姨+雨伞+老王)": lambda s: ("赵阿姨" in s and "雨伞" in s and "老王" in s),
        "C07_新组合(智能门锁+地下车库+机械钥匙)": lambda s: ("智能门锁" in s and "地下车库" in s and "机械钥匙" in s),
        "C08_新组合(随身听+磁带+修音轨)": lambda s: ("随身听" in s and "磁带" in s and "修音轨" in s),
        "C09_新组合(老花镜+微信+闹钟)": lambda s: ("老花镜" in s and "微信" in s and "闹钟" in s),
        "C10把被_新组合(周阿姨+陶瓷花瓶)": lambda s: ("周阿姨" in s and "陶瓷花瓶" in s),
        "C10归类_新组合(萤火虫+望远镜+鹈鹕)": lambda s: ("萤火虫" in s and "望远镜" in s and "鹈鹕" in s),
        "C10病句_新组合(书房+大概+二十平方+左右)": lambda s: ("书房" in s and "大概" in s and "二十" in s and "左右" in s),
        "1101定位_新组合(紫砂壶+双层隔热)": lambda s: ("紫砂壶" in s and "双层隔热" in s),
        "1101总结_新组合(手摇磨豆机+陶瓷磨芯)": lambda s: ("手摇磨豆机" in s and "陶瓷磨芯" in s),
        "1101基础_新组合(红皮萝卜+紫甘蓝)": lambda s: ("红皮萝卜" in s and "紫甘蓝" in s),
        "1101沅沅_新组合(猪脚饭+洒了一半)": lambda s: ("猪脚饭" in s and "洒了一半" in s),
    }
    hit_counts = {k: 0 for k in corpus_checks}
    total_rows = 0
    for p in sorted(glob.glob(os.path.join(RAW_DIR, "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                total_rows += 1
                for ck_k, fn in corpus_checks.items():
                    if fn(line):
                        hit_counts[ck_k] += 1
                for mk, kw in ROW_KEYWORDS.items():
                    if mk not in fetched and kw in line:
                        r = json.loads(line)
                        if ROW_MATCHERS[mk](r):
                            fetched[mk] = r

    print(f"全库扫描 {total_rows:,} 行完成 ({time.time()-t0:.1f}s) | 精确锁定训练原题 {len(fetched)}/{len(ROW_KEYWORDS)}", flush=True)
    if len(fetched) < len(ROW_KEYWORDS):
        missing = set(ROW_KEYWORDS) - set(fetched)
        raise RuntimeError(f"Missing target rows: {missing}")

    tk = Tokenizer(VOCAB)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = cfg_p4(**ck["cfg"])
    cfg["attn_backend"] = "auto"
    model = LMP4(cfg).float().eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})

    @torch.no_grad()
    def ask(q, ctx=None, max_new=110):
        text = "<bos>"
        if ctx:
            text += "<user>" + ctx + "<sep>"
        text += "<user>" + q + "<sep><assistant>"
        ids = torch.tensor([tk.encode(text)], dtype=torch.long)
        r = model.generate(
            ids, max_new_tokens=max_new, temperature=0.0, do_sample=False, top_k=1,
            eos_token_id=[tk.eos_id, tk.stoi["<sep>"]],
            repetition_penalty=1.08, no_repeat_ngram_size=4, context_len=ids.shape[1]
        )
        return tk.decode(r["new_tokens"], skip_special_tokens=True).strip()

    def row_qa(mk):
        r = fetched[mk]
        return r.get("context"), r["turns"][0]["text"], r["turns"][1]["text"]

    r1_ctx, r1_q, r1_a = row_qa("C01")
    r2_ctx, r2_q, r2_a = row_qa("C02")
    r3_ctx, r3_q, r3_a = row_qa("C03")
    r4_ctx, r4_q, r4_a = row_qa("C04")
    r5_ctx, r5_q, r5_a = row_qa("C05")
    r6_ctx, r6_q, r6_a = row_qa("C06")
    r7_ctx, r7_q, r7_a = row_qa("C07")
    r8_ctx, r8_q, r8_a = row_qa("C08")
    r9_ctx, r9_q, r9_a = row_qa("C09")
    r10a_ctx, r10a_q, r10a_a = row_qa("C10A")
    r10b_ctx, r10b_q, r10b_a = row_qa("C10B")
    r10c_ctx, r10c_q, r10c_a = row_qa("C10C")
    r11_ctx, r11_q, r11_a = row_qa("1101_LOC")
    r12_ctx, r12_q, r12_a = row_qa("1101_SUM")
    r13_ctx, r13_q, r13_a = row_qa("1101_BASE")
    r14_ctx, r14_q, r14_a = row_qa("1101_YUAN")

    c7_passage = r7_q.split("\n\n")[0]

    SUITES = [
        ("01. C01_故事叙事（按要素写故事）",
         (r1_ctx, r1_q, r1_a),
         (None, "请以山脚下清凉的溪边石桥为背景，写一个包含陈伯、一只耳廓狐，以及“困倦”状态的小故事。"),
         ("C01_新组合(陶艺+竹蜻蜓+琥珀)", None, "帮我写个小故事，人物用陶艺师阿周，地点是老街雨后的青石巷口，再加一只竹蜻蜓和一块琥珀，要带上惦记这个描述。")),

        ("02. C02_形容与颜色性状（词义辨析）",
         (r2_ctx, r2_q, r2_a),
         (None, "家里周末收拾时觉得东西又多又乱，形容这种房间状态该用“繁杂”还是“复杂”呀？"),
         ("C02_新组合(愤怒+手舞足蹈)", None, "“他愤怒地手舞足蹈，嘴角一直往上翘着。”我这么用愤怒这个词合适吗？")),

        ("03. C03_基础世界与动作（天然/人造辨别）",
         (r3_ctx, r3_q, r3_a),
         (None, "萝卜是从地里长出来的天然食物，那我刚买的这条皮带也是从土里长出来的吗？"),
         ("C03_新组合(登山袜+腐殖土)", None, "我刚买的这双尼龙登山袜，是不是跟野生蘑菇一样从树林腐殖土里长出来的天然物啊？")),

        ("04. C04_实物属性五层（食用属性纠偏）",
         (r4_ctx, r4_q, r4_a),
         (None, "在空心菜和四季豆之间，是不是空心菜必须完全熟透了吃，而四季豆吃前得剥掉外皮？"),
         ("C04_新组合(木陀螺+装水)", None, "买玩具的时候看到木陀螺，它到底是个什么东西，属于哪类常见物，能拿来当容器装水喝吗？")),

        ("05. C05_生活常识与概念（日期与常识纠偏）",
         (r5_ctx, r5_q, r5_a),
         (None, "每年国内的教师节到底是固定在公历的几月几号啊？我想给班主任发个祝福。"),
         ("C05_新组合(星期五天+双休日不算)", None, "别人跟我说一个星期其实只有五天，周六和周日双休日不算进星期里，这个说法对不对？")),

        ("06. C06_社会人文与角色（多人动作归属防混）",
         (r6_ctx, r6_q, r6_a),
         (None, "小区群看到“明天停水”消息后，沈青禾截图问慧姨真假，慧姨说要打电话向供水公司核实，林小满没核实就直接转进了业主群。请问是谁先提出打电话核实的，又是谁直接转发了？"),
         ("C06_新组合(赵阿姨+雨伞+老王)", None, "赵阿姨、小孙、老王三个人在地铁口躲雨，老王先把多余的一把雨伞借给了小孙，赵阿姨在旁边帮忙把湿雨衣叠好收进袋子。请问是不是赵阿姨把雨伞借给了小孙？")),

        ("07. C07_带资料定位与多轮（资料定位+防越界）",
         (r7_ctx, r7_q, r7_a),
         (None, c7_passage + "\n\n从上面这段银行营业厅资料看，地面适合用什么材料？能不能铺木地板？"),
         ("C07_新组合(智能门锁+地下车库+机械钥匙)", None, "请阅读下面这段资料：\n云杉小区地下车库本月新装了智能门锁，平时靠车牌识别或业主蓝牙自动抬杆；万一遇上停电故障，保安需要从值班室取出备用机械钥匙手动开锁。同时车库东入口常年备有二十只防汛沙袋应对暴雨倒灌。\n\n问题1：遇上停电故障时怎么手动开锁？另外这套门锁保修几年？")),

        ("08. C08_自我认知与边界（无跨会话记忆+纯文本边界）",
         (r8_ctx, r8_q, r8_a),
         (None, "上个礼拜咱俩聊过的那套方案你帮我回忆一下内容，原封不动再发我一遍呗？"),
         ("C08_新组合(随身听+磁带+修音轨)", None, "我家里那台老式磁带随身听卡带了，我拍了段录像发给你，你帮我看看怎么修一下音轨呗？")),

        ("09. C09_日常对话与情绪（多请求中挑能做的并拒越界）",
         (r9_ctx, r9_q, r9_a),
         (None, "我刚逛完菜市场回家，帮我做几件事吧：查一下这会儿外面下没下雨，顺便给明早六点设个闹钟，再帮我给老伴发微信说我买了青菜。"),
         ("C09_新组合(老花镜+微信+闹钟)", None, "我出门逛公园光顾着戴老花镜忘了看时间，你顺手帮我设个明早五点半的闹钟，再给我儿子发条微信让他来门口接我。")),

        ("10A. C10_句式转换（把字句 <-> 被字句）",
         (r10a_ctx, r10a_q, r10a_a),
         (None, "请把“杜小北把黄瓜放到了茶几上”改写成以“黄瓜”开头的“被”字句。"),
         ("C10把被_新组合(周阿姨+陶瓷花瓶)", None, "周阿姨把陶瓷花瓶摆在了窗台上。请把这句改成“被”字句。")),

        ("10B. C10_大小排序（四项比较与排序）",
         (r10b_ctx, r10b_q, r10b_a),
         (None, "把书桌、芝麻、篮球、苹果按体积从小到大排一下，哪个最小、哪个最大？"),
         ("C10归类_新组合(萤火虫+望远镜+鹈鹕)", None, "帮我把这四个归个类：萤火虫、望远镜、鹈鹕、车票。")),

        ("10C. C10_病句判断（有/无语病纠偏）",
         (r10c_ctx, r10c_q, r10c_a),
         (None, "帮我看看“无论天气多热，牛奶都得冷藏保存”这句话有没有语病，需要改吗？"),
         ("C10病句_新组合(书房+大概+二十平方+左右)", None, "“我家书房大概有二十平方米左右，平时挺安静的。”这句话有没有语病？帮我看看要不要改。")),

        ("11. 1101_定位训练（独立context短资料精准抽答）",
         (r11_ctx, r11_q, r11_a),
         (r11_ctx, "根据上面说的做法，做虎皮青椒挑的应该是哪一种辣椒？"),
         ("1101定位_新组合(紫砂壶+双层隔热)",
          "这款手工紫砂壶采用了双层隔热壶壁，端起来不容易烫手，平时专门用来冲泡乌龙茶以聚拢香气。壶嘴内侧还专门嵌入了一枚七孔瓷片滤网，用来防止细碎茶渣把出水口堵住。",
          "壶嘴里面用来防止茶渣堵住出水口的是什么结构？")),

        ("12. 1101_总结概括训练（独立context要点归纳）",
         (r12_ctx, r12_q, r12_a),
         (r12_ctx, "用简单的话概括一下压面器的主要用途和优点，顺便提一句清洗要注意啥。"),
         ("1101总结_新组合(手摇磨豆机+陶瓷磨芯)",
          "这款手摇磨豆机采用陶瓷磨芯，研磨时不容易发热，能更好留住咖啡豆原本的香气。机身配备六档粗细调节，手冲和摩卡壶都能适配。整机支持徒手拆洗，但洗完一定要彻底晾干再装回去，否则金属中轴容易生锈卡涩。",
          "这款手摇磨豆机主要有哪些特点，清洗后需要注意什么？")),

        ("13. 1101_基础知识问答（百科/食材外观对比）",
         (r13_ctx, r13_q, r13_a),
         (None, "把独头蒜和普通的蒜头放在一块儿，怎么从外形上一眼区分？"),
         ("1101基础_新组合(红皮萝卜+紫甘蓝)", None, "红皮萝卜和紫甘蓝放一起容易搞混吗？外形上有啥明显区别？")),

        ("14. 1101_沅沅对话（情绪接纳与共情追问）",
         (r14_ctx, r14_q, r14_a),
         (None, "上次房东突然通知不租了，我当着中介面强装没事笑呵呵的，可回到家越想越憋屈，算了不提了。"),
         ("1101沅沅_新组合(猪脚饭+洒了一半)", None, "真是倒霉透了，点的那份猪脚饭送到门口袋子破了，汤汁全洒在楼道里还烫了一下手，气死我了。")),
    ]

    out_lines = []
    def p(s=""):
        out_lines.append(s)
        print(s, flush=True)

    p("=" * 92)
    p("第 2 轮最佳模型（19.71M, val=1.8439）三层科学对照严格评测报告")
    p(f"全库扫描行数: {total_rows:,} | 全部采用 greedy (temperature=0.0, top_k=1) 无随机采样")
    p("=" * 92)

    for title, (a_ctx, a_q, a_gold), (b_ctx, b_q), (c_key, c_ctx, c_q) in SUITES:
        p(f"\n### {title}")
        p("-" * 88)
        ans_a = ask(a_q, ctx=a_ctx, max_new=110)
        ans_b = ask(b_q, ctx=b_ctx, max_new=110)
        ans_c = ask(c_q, ctx=c_ctx, max_new=110)

        p("【A. 训练集原题（确认已学过 2 遍）】")
        if a_ctx: p(f"  [CTX] {a_ctx}")
        p(f"  [问 ] {a_q}")
        p(f"  [标答] {a_gold}")
        p(f"  [实答] {ans_a}")

        p("【B. 同题改问法 / 变语序（测真理解 vs 死记硬背）】")
        p(f"  [问 ] {b_q}")
        p(f"  [实答] {ans_b}")

        p(f"【C. 未见过新组合（全库检索 `{c_key}` 命中 = {hit_counts[c_key]} 行）】")
        if c_ctx: p(f"  [CTX] {c_ctx}")
        p(f"  [问 ] {c_q}")
        p(f"  [实答] {ans_c}")

    p(f"\n总耗时: {time.time() - t0:.1f}s")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    p(f"已保存 -> {OUT_TXT}")


if __name__ == "__main__":
    main()
