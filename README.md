# MiniYuanYuan · 迷你沅沅

**在一张 RTX 3070 8GB 上，从零设计架构、自己处理语料、自己训完的中文对话模型。**
代码、230 万行对话语料、两版权重（FP16）全部在这个仓库里，没有一行是拿现成模型微调的。

| | 第 2 轮主线 | 第 3 轮 MoE |
|---|---|---|
| 唯一参数 | 19.71M | 30.34M（单 token 激活 23.26M） |
| 形状 | 12 层 / d384 / 12 Q-2 KV / 每 2 层共享 1 组 FFN | 同上，L6–L9 两组 FFN 升级成 MoE |
| 留出集 val | 1.8439（ppl 6.32） | **1.7744**（ppl 5.90） |
| 权重体积 | 45.8 MB（fp16） | 67.1 MB（fp16） |
| 4bit 估算 | ≈ 9.4 MiB | ≈ 14.5 MiB |

它是「沅沅」——湖南怀化沅陵县人设，会闲聊、会看资料答题、会拒绝它做不到的事。
但它**不是**小号 ChatGPT：30M 参数记不住冷门事实，也背不下课文。真实边界写在[已知局限](#八已知局限别拿它干这些)，建议先看那一节再看效果。

---

## 一、目录

```
MiniYuanYuan/
├── moxing/                 模型定义
│   ├── model_p4.py         第 1/2 轮：12 层 BiCrossGate（LMP4）
│   ├── model_moe.py        第 3 轮：FFN 组升级 MoE（LMMoE）
│   ├── model_arena.py      竞技场 3/5 号（标准 GQA 基线）
│   ├── model_1126.py       竞技场 1/2 号（4Q-NoKV 变体）
│   └── quant.py            QAT 伪量化（int4 权重 / int8 激活，STE）
├── tools/
│   ├── tokenizer.py        分词器（8302 词，字词混合）
│   ├── data.py             jsonl → packed bin（编码模板 + 打包）
│   ├── train_arena.py      统一训练入口（含 MoE 热升级）
│   ├── dashboard.py        本地训练监控网页（8765 端口）
│   └── eval_*.py           三套评测脚本（分片 CE / 三层对照 / 终版对比）
├── configs/                7 份架构配置（含竞技场 5 个对照）
├── corpus/
│   ├── raw/                24 个分片，参与过训练（旧语料）
│   ├── new/                2 个分片，训练之后才产出，没进过训练（新语料）
│   ├── samples/            每线前 100 条明文，先看再下
│   ├── vocab_v2.json       词表
│   └── MANIFEST.json       逐文件行数/字符数/压缩体积
├── weights/                两版 fp16 权重（见 weights/README.md）
├── docs/architecture.md    两代架构逐项对照 + 参数账 + MoE 误判复盘
├── docs/corpus.md          语料逐分片明细、格式、已知质量问题
├── docs/results.md         22 分片 CE 全表 + 能力实测（含失败的测试）
├── examples/chat.py        最小可运行对话示例
└── LICENSE                 MIT
```

## 二、先分清「旧 / 新」这两个词

仓库里「新」「旧」有两种含义，容易混：

| 说法 | 指什么 | 在哪 |
|---|---|---|
| **旧项目** | `1124` 那条弯路：24.87M，36 个计算层，18 套 Attn 跑两遍，MTP3，val 2.3267 | 代码**不发布**，只在 `docs/architecture.md` 里记录形状与放弃原因 |
| **新项目** | `1125_jingjichang`：19.71M / MoE 30.34M，12 层每 2 层共享 FFN，val 1.8439 / 1.7744 | **本仓库主体**，代码 + 配置 + 权重全在 |
| **旧语料** | `corpus/raw/` 24 个分片 | 就是两版权重真正训练过的数据 |
| **新语料** | `corpus/new/` 2 个分片（C13 数学换算、SG 故事概括） | 训练结束之后才生成，**没进过任何一轮** |

> 所以：**两版权重都没有在新语料上训练过。** 想复现 1.7744 请用 `corpus/raw`；想更强，把 `new` 并进去重训。

## 三、快速开始

只要 PyTorch 2.x，CPU 也能跑（单条回复 1–2 秒），有 CUDA 会自动用。

```bash
pip install torch

# 默认加载第 3 轮 MoE（30.34M，效果最好）
python examples/chat.py --prompt "你好，你是谁？"

# 带资料提问（模拟 RAG 定位，这是它最稳的能力）
python examples/chat.py --context "沅陵县位于湖南省怀化市西北部，沅水中游。" --prompt "沅陵县在哪个市？"

# 换 19.71M 那版
python examples/chat.py --model weights/m12l_d384_ep2_fp16.pt --prompt "把这句话改通顺：我家客厅三十平方米。"

# 采样模式（默认 temp=0 贪心解码，确定性输出）
python examples/chat.py --prompt "讲个短故事" --temp 0.8 --max-new 200
```

实测输出（`temp=0`，可复现）：

```
模型：m12l_d384_moe_ep3_fp16.pt（step 17000，best_val 1.7744，dtype float16）
------------------------------------------------------------
我是沅沅，来自湘西龙舟之乡沅陵县。
```

对话模板（`max_len = 1088`，自己写推理时照这个拼）：

```
单轮      <bos><user>{问题}<sep><assistant>
多轮      <bos><user>{问1}<sep><assistant>{答1}<sep><user>{问2}<sep><assistant>
带资料    <bos><user>{资料}<sep><user>{问题}<sep><assistant>
停止符    eos_id 与 <sep>
```

## 四、语料

**230.9 万行 / 2.96 亿字正文**，来自两条自建产线：`1101_新主题` 占 61.1%、`1123` 占 38.9%（按正文字数）。
全部是 API 批量生成 + 规则构造的合成对话，统一成 `{"id": ..., "turns": [{"role","text"}...]}`，lzma 压到 244 MB（解开约 1.5 GB）。
逐分片行数、正文字数、来源与质量问题见 **[docs/corpus.md](docs/corpus.md)**。

- 先看 `corpus/samples/*.jsonl`（每线 100 条明文）判断要不要下全量；
- 全量在 `corpus/raw/*.xz` 与 `corpus/new/*.xz`；
- 行数、字符数、压缩前后体积见 `corpus/MANIFEST.json`。

| 线 | 行数 | 内容 |
|---|---:|---|
| 1101__基础知识型问答 | 917,481 | 百科式问答，最大的一坨 |
| 1123__C04_实物属性五层 | 275,624 | 物品属性分层描述 |
| 1101__常识性一问一答 | 195,737 | 短问答 |
| 1101__定位训练_提供资料_多轮提问 | 192,668 | RAG 定位（长资料） |
| 1101__定位训练_日常对话中的定位 | 163,967 | RAG 定位（对话里夹资料） |
| 1123__C07_带资料定位与多轮 | 87,483 | RAG 定位 |
| new/C13_数学换算与比较 | 70,576 | **未进训练** |
| 1123__C10_文字改写与语言任务 | 63,491 | 改写 / 病句 / 句式转换 |
| 1123__C03_基础世界与动作 | 60,994 | 世界常识与动作 |
| 1123__C09_日常对话与情绪 | 53,838 | 情绪安抚、闲聊 |
| 1101__总结概括训练_多轮概括 | 40,134 | 摘要 |
| 1123__C02_形容与颜色性状 | 38,498 | 形容词辨析 |
| 1101__沅沅对话 | 28,050 | 人设对话 |
| 1101__常识问答 | 25,457 | 短问答 |
| new/SG_故事概括与多维组合写作 | 20,579 | **未进训练** |
| 1123__C05_生活常识与概念 | 20,298 | 生活常识 |
| 1123__C08_自我认知与边界 | 13,883 | 拒答、能力边界 |
| 1123__C01_故事叙事 | 12,577 | 叙事 |
| 1101__新主题长资料_A批 | 11,240 | 长资料 |
| 1123__C06_社会人文与角色 | 11,038 | 社会角色 |
| 1101__日常寒暄与闲聊 | 2,835 | 寒暄 |
| 1101__世界历史、中国历史 | 1,281 | 历史 |
| 1101__物理化学 | 447 | 理化 |
| 1101__自我介绍 | 310 | 自我介绍 |
| 1101__人物 | 237 | 人物 |
| 1101__现代生活制度 | 111 | 制度常识 |
| **训练用 24 线合计** | **2,217,679** | 2.80 亿字 |
| **新语料 2 线合计** | **91,155** | 0.16 亿字，未进训练 |

语料的真实水平：模板化明显、同义句重复多、事实密度低。它够让 30M 模型学会「中文对话长什么样」，不够让它变成知识库。别把这批数据当高质量语料库用。

## 五、架构

完整参数账、每层形状、竞技场 5 个对照模型的成绩、以及第三轮 MoE 的复盘都在 **[docs/architecture.md](docs/architecture.md)**，这里只留结论。

```
token → Embedding(8302×384)
      → 12 × [ RMSNorm → GQA(12Q/2KV, hd=32) → 残差 → RMSNorm → 共享 FFN → 残差 ]
      → RMSNorm → lm_head（与 Embedding 绑权重）
```

- **每 2 层共享一组 FFN**（`ffn_share_group=2`，共 6 组）：注意力、两个 RMSNorm 各层独立，只有 FFN 共享。这是最狠的一刀，也是后来想拆开的原因。
- **BiCrossGateFFN**：`gate_proj(cat(act1(q1*x), act2(q2*x))) * up_proj(x)`，`2d²+4di`。竞技场里同参数量下打赢了标准 SwiGLU（val 2.5505 vs 2.5761）。
- **参数分布**：FFN 12.39M（62.8%）、Attention 4.13M（20.9%）、Embedding 3.19M（16.2%）。所以「共享 FFN」省的正是大头。

### 第三轮 MoE 是一次误判，写在这里当反面教材

当时想解决的问题是「FFN 容量被共享压得太狠」。正确的改法只有一个变量：把 `ffn_share_group` 从 2 改成 1，12 层各给一组独立 FFN（32.10M）。实际做的却是把两组 FFN 升级成 MoE（30.34M，6 专家 Top-2 + 无辅助损失偏置均衡）。

参数量几乎一样（32.10M vs 30.34M），但 MoE 同时动了路由、专家分配、负载均衡、稀疏梯度四套机制，训完**没法回答**「收益到底来自容量还是来自稀疏」。MoE 版确实把 val 从 1.8439 压到 1.7744（22 个留出分片 CE 全降），但它回答不了当初那个问题，还多了一堆调参负担。方法论错误，不是实现错误。

想拆共享，改配置更省事：

```json
{ "ffn_share_group": 1 }
```

只动这一个字段，结构、训练目标、优化器全不变，训完可以直接归因「就是容量变大了」。

## 六、训练

单卡 RTX 3070 8GB，bf16，无梯度检查点。

| | 第 2 轮（主线） | 第 3 轮（MoE 热升级） |
|---|---|---|
| 步数 | 34,000（2 epoch） | 17,000（1 epoch，从第 2 轮 best 续） |
| batch | 8 × accum 2 × 1088 = 17,408 tok/step | 同左 |
| 学习率 | 4e-4，cosine → 5%，warmup 300 | 2e-4，MoE 参数 ×2 |
| 其他 | wd 0.1 / clip 1.0 / seed 1337 | `moe_down_scale 0.05` / seed 21351 |
| 训练 token | 2.96 亿（packed，loss 计入 1.52 亿） | 同 |
| 吞吐 | ≈ 49.7k tok/s | ≈ 39.2k tok/s |
| 峰值显存 | 3.89 GB | 4.98 GB |
| 墙钟 | 约 2 小时 02 分 | 约 2 小时 05 分 |
| val | **1.8439** | **1.7744** |

复现主线（先用 `corpus/samples` 跑通，再换全量）：

```bash
# 1) 解压 + 打包（仓库不放 packed bin：单文件 565 MB，超 GitHub 100 MiB 限制）
for f in corpus/raw/*.jsonl.xz; do xz -dk "$f"; done        # 保留 .xz，解出 .jsonl
python tools/data.py --src corpus/raw --out dataset/full --max-len 1088
#    想先小跑一遍：--src corpus/samples （每线 100 条，几十秒打包完）
#    val 切分：id 以 val- 开头，或按 id 的 md5 取模取 0.5%（--val-ratio 可调）

# 2) 训练 12 层 19.71M
python tools/train_arena.py \
  --config configs/m12l_d384_stdgqa_kv2_bicrossgate1152_share2_w1088.json \
  --data dataset/full --run m12l_ep2 --out out \
  --batch 8 --grad-accum 2 --max-len 1088 --lr 4e-4 --epochs 2 --dtype bf16

# 3) MoE 热升级（从第 2 轮 best 续训；也可以直接拿仓库里那版权重当起点）
python tools/train_arena.py \
  --config configs/m12l_d384_moe_g34_e6k2_w1088.json \
  --data dataset/full --run m12l_moe --out out \
  --init-upcycle out/m12l_ep2/best.pt --lr 2e-4 --epochs 1 \
  --moe-lr-mult 2.0 --moe-down-scale 0.05
#    --init-upcycle weights/m12l_d384_ep2_fp16.pt 也行（会从 fp16 权重热升级）
```

`tools/data.py` 吃的是 `{"id","turns"}` 的 jsonl；`--src` 给目录会按文件名排序全部读入。
没有 token 缓存（`dataset/*_cache`）时训练会自动跳过「每轮重新打散」，直接用打包好的 bin —— 想开这个功能先跑 `tools/data.py` 里的 `build_cache` 入口或 `tools/build_full_1125.py`。
训练曲线用 `python tools/dashboard.py --out out` 起本地网页看（8765 端口）。

## 七、实测能力（诚实版）

留出集上 22 个分片全部优于上一轮，但 **loss 低 ≠ 会答题**（逐分片全表与测试细节见 **[docs/results.md](docs/results.md)**）：

| 现象 | 数据 |
|---|---|
| 分片 CE（MoE 版） | 最好的 C07 带资料定位 **0.209**；最差的 1101__人物 **3.231** |
| 45 题人工判分（旧 1124 版） | 全对 3/45，半对 4 题；失败模式几乎都是「主题词 + 模板拼接」 |
| 开卷 vs 闭卷复现（旧 1124 版） | 开卷逐字命中 36.8% → 闭卷 4.0%（≈ 随机），记的是语言形状不是内容 |

**它擅长的**（不同测试里都稳定）：

1. **看资料抽取**：给一段没见过的资料提问，能准确抓出答案（C07 留出 CE 0.209，最低的一档）；
2. **拒答与边界**：联网查天气、设闹钟、发微信、跨会话记忆这类请求，会明确说做不到并给替代方案；
3. **日常闲聊与情绪安抚**：人设稳定，语气自然；
4. **部分语言规则改写**：把字句 / 被字句互换能泛化到没见过的句子。

**它不擅长的**：冷门百科数值与日期（会串实体）、三个人物的多步动作追踪、没见过的算术、病句判断（默认倾向「没有语病」）、任何需要精确背诵的内容。

## 八、已知局限（别拿它干这些）

- **不要当知识库**：参数里装不下事实，问细节大概率编。要准就上 RAG，它的定位能力比记忆能力高两个数量级。
- **不要用于医疗、法律、金融等决策**，也不要用它的输出当事实依据。
- **不要指望长文**：训练时 `max_len=1088`，超过这个长度质量断崖。
- **不要用 fp16 权重做逐位复现**：与 fp32 原权重有 0.1%–0.3% 的概率分歧（见 `weights/README.md` 的对照表）。
- 语料是合成的，含模板与重复，模型会复述这些模板腔。

## 九、许可

代码 MIT（见 `LICENSE`）。语料为自建合成数据，随仓库以 MIT 发布。权重同 MIT。
模型输出不保证正确，使用者自担风险。

---

## English

**MiniYuanYuan** is a Chinese chat model trained **from scratch** on a single consumer GPU (RTX 3070, 8 GB) — no pretrained base, no distillation, no borrowed weights.

- **Two checkpoints**: 19.71M params (12 layers, d384, GQA 12Q/2KV, one shared BiCrossGate FFN per 2 layers) and a 30.34M-param MoE variant (23.26M active) that upgrades two FFN groups to 6 experts with top-2 routing.
- **Held-out loss**: 1.8439 (ppl 6.32) and **1.7744** (ppl 5.90). All 22 held-out shards improve in the MoE version.
- **Corpus**: 2.31M synthetic Chinese dialogue rows (~296M characters), 24 shards used for training plus 2 shards generated *after* training finished (never seen by either checkpoint). Compressed to 244 MB with lzma; 100-line plaintext samples per shard are included so you can inspect before downloading.
- **Good at**: reading comprehension over a supplied passage (RAG-style grounding), refusing out-of-scope requests, casual chit-chat, and a few syntactic rewriting rules.
- **Cannot do**: recall obscure facts, multi-step reasoning, or verbatim memorization. At 30M parameters it learns the *shape* of Chinese, not its content. Do not use it as a knowledge base or for medical/legal/financial decisions.

```bash
pip install torch
python examples/chat.py --prompt "你好，你是谁？"
# -> 我是沅沅，来自湘西龙舟之乡沅陵县。
```

Training recipe, architecture comparison, and a post-mortem of the third-round MoE decision are in [`docs/architecture.md`](docs/architecture.md). Everything here is MIT licensed.
