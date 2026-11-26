# MiniYuanYuan · 迷你沅沅

**一套从零自建的 230.9 万行 / 约 2.96 亿字中文小模型对话语料库（含约 33% 拒答/纠偏/澄清负样本），以及在单卡 RTX 3070 8GB 上验证可用的参考基线模型与训练流程。**

> **核心定位说明**：本仓库**最核心的资产是「语料库与语料配方」**，而不是某个非用不可的魔改模型结构。
> 如果你只是想拿这套语料去训小模型，**直接用传统的标准 Transformer（如经典 Llama / GQA + SwiGLU）也完全可以**。作者本人因为没有多余时间和算力去反复对比各种架构，仓库里放出的 19.71M / 30.34M 模型与训练脚本，主要用于**验证这套语料能训通、能学会中文对话与看资料回答**，给你提供一套开箱可跑的基线参考与完整训练过程记录。

| 仓库核心内容 | 规模 / 说明 | 位置 |
|---|---|---|
| **开源中文对话语料（主力）** | **230.9 万行 / ~2.96 亿字**（含 24 个已训分片 + 2 个未训新分片） | `corpus/`、[docs/corpus.md](docs/corpus.md)、[docs/corpus_generation.md](docs/corpus_generation.md) |
| **参考分词器与打包训练代码** | 8,302 字词混合词表，完整 PyTorch 训练/评测脚本 | `tools/`、`moxing/`、`configs/` |
| **两版基线权重（FP16）** | 第 2 轮主线 19.71M（val 1.8439） / 第 3 轮 MoE 30.34M（val 1.7744） | `weights/`、[docs/architecture.md](docs/architecture.md) |

它是「沅沅」——湖南怀化沅陵县人设，会闲聊、会看资料答题、会拒绝它做不到的事。但 20M～30M 小模型记不住冷门事实，真实边界写在[已知局限](#六已知局限别拿它干这些)。

---

## 一、语料的组成（核心重点）

整套语料共 **2,308,834 行 / 约 2.96 亿字正文**（不含 JSON 结构字符），统一为 `{"id": ..., "turns": [{"role":"user"|"assistant","text":...}]}`（纯文本分片带 `text` 字段），经 `lzma` 压缩后约 244 MB（解压约 1.5 GB）。

语料由两条自建 API 产线组成：
- **`1101` 产线（占已训语料正文字数 61.1%）**：偏重基础百科问答、常识问答、大批量 RAG 带资料定位与「沅沅」人设对话，负责把小模型的中文基础表达与资料抽取能力铺厚；
- **`1123` 产线（占已训语料 38.9% + 全部未训新语料）**：基于 **3,196 个核心一级词 × 10 张结构化母表** 正交调度，并用三层确定性脚本门禁清洗（实测通过率 **98.67%**），重点补齐**实物属性、动作逻辑、句式改写、自我边界以及高比例负样本**。

### 1.1 按能力板块划分的语料组成

| 能力板块 | 包含分片 | 行数 | 核心作用与设计特点 |
|---|---|---:|---|
| **1. 基础世界与百科常识** | `1101__基础知识型问答`、`1101__常识性一问一答`、`1101__常识问答`、`1123__C03_基础世界与动作`、`1123__C05_生活常识与概念`、`1123__C06_社会人文与角色`、`1101__世界历史/物理化学/人物/制度` | 1,232,913 | 让小模型建立日常世界、物理现象、社会角色与常见概念的中文语义底座 |
| **2. 实物属性与形容辨析** | `1123__C04_实物属性五层`、`1123__C02_形容与颜色性状` | 314,122 | 按颜色/材质/用途/结构/交互五层拆解常见物品，强化名词—属性—形容词绑定 |
| **3. 看资料精准定位（RAG）与概括** | `1101__定位训练_提供资料_多轮提问`、`1101__定位训练_日常对话中的定位`、`1123__C07_带资料定位与多轮`、`1101__总结概括训练`、`1101__新主题长资料_A批` | 495,492 | **小模型最稳的核心能力**：把资料放前面、多轮紧扣原文细节提问（含“文中未提就拒答”），训练只据原文回答、不瞎编 |
| **4. 日常对话、情绪与人设** | `1101__沅沅对话`、`1101__日常寒暄与闲聊`、`1101__自我介绍`、`1123__C09_日常对话与情绪` | 85,033 | 寒暄、道谢道歉、情绪安慰、换题澄清，塑造稳定自然的「沅沅」口语对话感 |
| **5. 文字改写与语言规则** | `1123__C10_文字改写与语言任务` | 63,491 | 把字句/被字句互换、改病句、排序、归类、格式约束与多步指令 |
| **6. 能力边界与故事叙事** | `1123__C08_自我认知与边界`、`1123__C01_故事叙事` | 26,460 | 拒答越界请求（联网查实时天气、设闹钟、跨会话记忆）+ 全第三人称日常故事 |
| **7. 高比例负样本（贯穿 1123 产线，约占 33%）** | 分布于 `C01`～`C10`、`C13`、`SG` 各轨道 | （内嵌于各分片） | 包含三大类：**错答认错纠偏**（首轮错答不计 Loss，仅终轮纠偏计 Loss）、**未提拒编**（文中没写就说不知道）、**缺条件先澄清 / 越界拒答** |
| **8. 未进过当前权重训练的新语料（`corpus/new/`）** | `C13_数学换算与比较`、`SG_故事概括与多维组合写作` | 91,155 | 训练结束后才产出，**两版公开权重均未见过**：含 7 万条分步数学应用题（含缺条件拒算）与 2 万条 600 字以内多维组合写故事/多轮反问改稿 |

### 1.2 逐分片数量速览与查看方式

- **先看样本再下全量**：`corpus/samples/*.jsonl` 收录了每个分片前 100 条明文；
- **全量压缩包**：`corpus/raw/*.xz`（24 个已训分片，221.8 万行）与 `corpus/new/*.xz`（2 个未训新分片，9.1 万行）；
- **详细产出流程、Prompt 长什么样、以及尚缺最后 10 包待补清单**：见 **[docs/corpus_generation.md](docs/corpus_generation.md)**；
- **逐分片字数与已知瑕疵说明**：见 **[docs/corpus.md](docs/corpus.md)**（注：当前已训版本存在一定模板化与同义重复，适合训练中文对话骨架与 RAG 定位，不建议当百科事实库；如要继续提高能力，建议将 `corpus/new/` 并入重训，并对 `C04` 适当降权或去重）。

<details>
<summary>展开查看全部 26 个语料分片行数对照表</summary>

| 线 | 行数 | 内容 |
|---|---:|---|
| 1101__基础知识型问答 | 917,481 | 百科式问答 |
| 1123__C04_实物属性五层 | 275,624 | 物品属性分层描述 |
| 1101__常识性一问一答 | 195,737 | 短问答 |
| 1101__定位训练_提供资料_多轮提问 | 192,668 | RAG 定位（长资料） |
| 1101__定位训练_日常对话中的定位 | 163,967 | RAG 定位（对话里夹资料） |
| 1123__C07_带资料定位与多轮 | 87,483 | RAG 定位 + 未提拒答 |
| new/C13_数学换算与比较 | 70,576 | **未进训练**（数学 + 缺条件拒算） |
| 1123__C10_文字改写与语言任务 | 63,491 | 改写 / 病句 / 句式转换 |
| 1123__C03_基础世界与动作 | 60,994 | 世界常识与动作 |
| 1123__C09_日常对话与情绪 | 53,838 | 情绪安抚、闲聊 |
| 1101__总结概括训练_多轮概括 | 40,134 | 摘要 |
| 1123__C02_形容与颜色性状 | 38,498 | 形容词辨析 |
| 1101__沅沅对话 | 28,050 | 人设对话 |
| 1101__常识问答 | 25,457 | 短问答 |
| new/SG_故事概括与多维组合写作 | 20,579 | **未进训练**（≤600字故事 + 多轮反问改稿） |
| 1123__C05_生活常识与概念 | 20,298 | 生活常识 |
| 1123__C08_自我认知与边界 | 13,883 | 拒答、能力边界 |
| 1123__C01_故事叙事 | 12,577 | 第三人称叙事 |
| 1101__新主题长资料_A批 | 11,240 | 长资料 |
| 1123__C06_社会人文与角色 | 11,038 | 社会角色 |
| 1101__日常寒暄与闲聊 | 2,835 | 寒暄 |
| 1101__世界历史、中国历史 | 1,281 | 历史 |
| 1101__物理化学 | 447 | 理化 |
| 1101__自我介绍 | 310 | 自我介绍 |
| 1101__人物 | 237 | 人物 |
| 1101__现代生活制度 | 111 | 制度常识 |
| **训练用 24 线合计** | **2,217,679** | 约 2.80 亿字 |
| **新语料 2 线合计** | **91,155** | 约 1,642 万字，未进训练 |

</details>

---

## 二、仓库目录结构

```text
MiniYuanYuan/
├── corpus/
│   ├── raw/                  24 个分片，参与过当前两版权重训练
│   ├── new/                  2 个新分片（C13 数学、SG 故事），训练后产出、未进训练
│   ├── samples/              每线前 100 条明文，先看再下
│   ├── vocab_v2.json         词表（8,302 字词混合）
│   └── MANIFEST.json         逐文件行数/字符数/压缩体积
├── docs/
│   ├── corpus.md             语料逐分片明细、格式、已知质量问题
│   ├── corpus_generation.md  1123 线语料产出流程、提示词样例与待补全清单
│   ├── architecture.md       基线架构参数账与 MoE 复盘（仅供结构参考）
│   └── results.md            22 分片 CE 全表 + 能力实测（含失败测试）
├── tools/
│   ├── tokenizer.py          分词器（8,302 词）
│   ├── data.py               jsonl → packed bin（编码模板 + 打包）
│   ├── train_arena.py        统一训练入口（含 MoE 热升级）
│   ├── dashboard.py          本地训练监控网页（8765 端口）
│   └── eval_*.py             三套评测脚本
├── moxing/                   模型定义（BiCrossGate / MoE / 标准 GQA 对照等）
├── configs/                  架构配置文件
├── weights/                  两版 fp16 参考权重（见 weights/README.md）
├── examples/chat.py          最小可运行对话示例
└── LICENSE                   CC BY-NC 4.0（仅限非商业使用）
```

---

## 三、我的架构选择与实际训练过程（给复现/自训者的参考）

> **再次说明**：如果你拿这套语料去训自己的模型，**完全没必要照搬这里的 `BiCrossGate` 或 `MoE`**，直接用最省事的**标准传统 Transformer（RoPE + RMSNorm + GQA/MHA + SwiGLU）**就足够了，效果同样稳、还少很多调参包袱。
> 作者之所以用下面这套结构，只是在 RTX 3070 8GB 有限显存下做的一次快速验证，没有时间再拆开做多轮架构消融实验。

### 3.1 作者实际跑过的架构形状

| | 第 2 轮主线（基线） | 第 3 轮 MoE（续训） | 如果你用传统 Transformer 怎么选 |
|---|---|---|---|
| 唯一参数 | 19.71M | 30.34M（单 token 激活 23.26M） | 直接设 12 层 `d=384` 标准 SwiGLU（约 25M～32M）即可 |
| 网络形状 | 12 层 / d384 / 12Q-2KV / 每 2 层共享 1 组 BiCrossGate FFN | 同上，L6–L9 两组 FFN 升级为 6 专家 Top-2 MoE | 标准 12 层独立 FFN（不必共享、不必上 MoE） |
| 留出集 val | 1.8439（ppl 6.32） | **1.7744**（ppl 5.90） | — |
| 权重体积 | 45.8 MB（fp16） | 67.1 MB（fp16） | — |

```text
token → Embedding(8302×384)
      → 12 × [ RMSNorm → GQA(12Q/2KV, hd=32) → 残差 → RMSNorm → 共享 BiCrossGate FFN → 残差 ]
      → RMSNorm → lm_head（与 Embedding 绑权重）
```

> 详细的每一层参数账与作者在第三轮 MoE 上的事后复盘（为什么原本只需把 `ffn_share_group` 从 `2` 改成 `1` 即可、没必要引入 MoE），见 **[docs/architecture.md](docs/architecture.md)**。

### 3.2 实际训练过程（单卡 RTX 3070 8GB · bf16 · 无梯度检查点）

作者完整训练一共走了两阶段（总计约 4 小时 07 分）：

1. **阶段 1 — 语料编码打包**：用 `tools/data.py` 把 `corpus/raw/` 的 24 个分片按 `max_len = 1088` 打包成训练用二进制（221.8 万行 → 27.2 万窗口 / **2.96 亿 token**，其中助手回复计入 Loss 约 **1.52 亿 token**）。
2. **阶段 2 — 第 2 轮主线从零训练（34,000 步 / 2 Epoch）**：
   - batch `8 × accum 2 × 1088 = 17,408 tok/step`，学习率 `4e-4`（warmup 300 步 + cosine 衰减至 5%），`wd=0.1`、`clip=1.0`。
   - 吞吐约 `49.7k tok/s`，峰值显存 `3.89 GB`，耗时 **约 2 小时 02 分**，留出集 `val loss` 收敛到 **1.8439**。
3. **阶段 3 — 第 3 轮 MoE 热升级续训（17,000 步 / 1 Epoch）**：
   - 直接从第 2 轮 `best.pt` 权重起步，只把中间 L6–L9 两组 FFN 热升级为 MoE 续训 1 个 Epoch（学习率降为 `2e-4`，MoE 参数 ×2）。
   - 吞吐约 `39.2k tok/s`，峰值显存 `4.98 GB`，耗时 **约 2 小时 05 分**，留出集 `val loss` 进一步压到 **1.7744**。

---

## 四、快速开始（直接跑权重或自己拿语料重训）

### 4.1 体验现成权重

只要 PyTorch 2.x，CPU 也能跑（单条回复 1–2 秒），有 CUDA 会自动用：

```bash
pip install torch

# 默认加载第 3 轮 MoE（30.34M，效果最好）
python examples/chat.py --prompt "你好，你是谁？"

# 带资料提问（模拟 RAG 定位，这是它最稳的能力）
python examples/chat.py --context "沅陵县位于湖南省怀化市西北部，沅水中游。" --prompt "沅陵县在哪个市？"

# 换 19.71M 主线版
python examples/chat.py --model weights/m12l_d384_ep2_fp16.pt --prompt "把这句话改通顺：我家客厅三十平方米。"
```

对话模板（`max_len = 1088`，自己写推理时照这个拼）：

```text
单轮      <bos><user>{问题}<sep><assistant>
多轮      <bos><user>{问1}<sep><assistant>{答1}<sep><user>{问2}<sep><assistant>
带资料    <bos><user>{资料}<sep><user>{问题}<sep><assistant>
停止符    eos_id 与 <sep>
```

### 4.2 用这套语料从零复现或训练你自己的 Transformer

```bash
# 1) 解压 + 打包（如果想把新语料 corpus/new/ 也一起训进去，先合并到同一个目录再打包）
for f in corpus/raw/*.jsonl.xz; do xz -dk "$f"; done
python tools/data.py --src corpus/raw --out dataset/full --max-len 1088
#    想先小跑几十秒验流程：--src corpus/samples

# 2) 从零训练 12 层 19.71M 基线（也可以换成 moxing/model_arena.py 里的标准 GQA+SwiGLU）
python tools/train_arena.py \
  --config configs/m12l_d384_stdgqa_kv2_bicrossgate1152_share2_w1088.json \
  --data dataset/full --run m12l_ep2 --out out \
  --batch 8 --grad-accum 2 --max-len 1088 --lr 4e-4 --epochs 2 --dtype bf16

# 3) 可选：MoE 热升级续训 1 epoch
python tools/train_arena.py \
  --config configs/m12l_d384_moe_g34_e6k2_w1088.json \
  --data dataset/full --run m12l_moe --out out \
  --init-upcycle out/m12l_ep2/best.pt --lr 2e-4 --epochs 1 \
  --moe-lr-mult 2.0 --moe-down-scale 0.05
```

> 本地训练曲线监控：`python tools/dashboard.py --out out`（8765 端口）。

---

## 五、实测能力与边界（诚实版）

留出集 22 个分片 CE 全表与测试细节见 **[docs/results.md](docs/results.md)**：
- **它擅长的**：① **看资料抽取**（C07 留出 CE **0.209**，最稳的一项）；② **拒答与边界**（联网查天气、设闹钟、跨会话记忆会明确拒答并给替代建议）；③ **日常闲聊与情绪安抚**；④ **部分语言规则改写**（把字句/被字句等）。
- **它不擅长的**：冷门百科数值与日期（会串实体）、三个人物多步动作追踪、没见过的算术（当前权重未训 `corpus/new/C13`）、病句判断、精确背诵课文。

## 六、已知局限（别拿它干这些）

- **不要当知识库**：30M 参数里装不下冷门事实，问细节大概率编；要准请走 RAG（给资料再问）。
- **不要用于医疗、法律、金融等决策**，也不要用它的输出当事实依据。
- **不要指望长文**：训练时 `max_len=1088`，超过此长度质量明显下降。
- **不要用 fp16 权重做逐位复现**：与 fp32 原权重存在 `0.1%–0.3%` 概率分歧（见 `weights/README.md`）。

## 七、先分清文档里的「旧 / 新」

| 说法 | 指什么 | 在哪 |
|---|---|---|
| **旧语料** | `corpus/raw/` 24 个分片（221.8 万行） | 两版权重真正训练过的数据 |
| **新语料** | `corpus/new/` 2 个分片（`C13` 数学 + `SG` 故事，9.1 万行） | 训练结束后才产出，**两版权重都没训过** |
| **旧项目（1124）** | 早期 24.87M / 36 层微循环弯路（val 2.3267） | 仅在 `docs/architecture.md` 记录，不发布 |
| **新项目（1125）** | 当前 19.71M / MoE 30.34M 基线（val 1.8439 / 1.7744） | 本仓库代码与权重 |

## 八、许可

本仓库全部内容（代码、自建合成语料、模型权重）均采用 **CC BY-NC 4.0（署名-非商业性使用 4.0 国际许可协议）** 发布（详见 [LICENSE](LICENSE)）：

- **允许**：个人自用、学习研究、修改、二次训练与非商业署名分享；
- **禁止**：任何商业用途（不得直接或间接拿去售卖、接入收费服务、用于商业产品或商业闭源训练盈利）；如需商用请联系作者另行授权。

模型输出不保证正确，使用者自担风险。

---

## English

**MiniYuanYuan** is an open-source synthetic Chinese dialogue corpus (**2.31M rows / ~296M characters**, including ~33% refusal/correction/clarification negative samples), accompanied by lightweight reference checkpoints (19.71M & 30.34M MoE) trained from scratch on a single RTX 3070 (8 GB) to validate the dataset.

- **Corpus-first repository**: The dataset and data-generation pipeline ([`docs/corpus_generation.md`](docs/corpus_generation.md)) are the core focus. You do **not** have to use the custom `BiCrossGate`/MoE architecture in this repo — a **standard Transformer (e.g., GQA + SwiGLU)** works equally well on this corpus. The included checkpoints simply serve as a reproducible baseline trained under limited time and single-GPU compute.
- **Corpus composition**: Built from two synthetic lines — `1101` (61.1% of training chars: encyclopedic QA, general chit-chat, persona, and large-scale RAG grounding) and `1123` (38.9% of training chars + all unseen new shards: 3,196 core vocabulary items across object properties, world/action, text rewriting, self-boundaries, plus `~33%` negative samples). Includes 24 trained shards (`corpus/raw/`) and 2 post-training shards (`corpus/new/`: `C13` math & `SG` <=600-char compositional stories).
- **Training recipe (reference baseline)**: Stage 1 packs `corpus/raw/` at `max_len=1088` (2.96亿 tokens total, 1.52亿 loss tokens); Stage 2 trains the 19.71M base for 2 epochs (34k steps, ~2h 02m, val 1.8439); Stage 3 upcycles two FFN groups to 6-expert top-2 MoE for 1 epoch (17k steps, ~2h 05m, val **1.7744**).
- **License**: Everything (code, corpus, and weights) is licensed under **CC BY-NC 4.0** (free for personal/educational/non-commercial use with attribution; commercial use prohibited without separate written permission).
