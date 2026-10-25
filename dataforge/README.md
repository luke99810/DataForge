# DataForge 编排层

把 `dataset_generator`（确定性内核）包成一个可观测、可重放、可评测的 Agent Harness。

设计依据：桌面上的《DataForge-设计文档.html》。已实现 **M0 地基**、**M1 Harness**、
**M2 桌面壳**、**M3 论文链路**、**Refine 外层循环**、**Planner**、**MCP 服务端**、
**RL / LLM 训练集 / Agent 测评集生成器**、**外部源对齐**。

```bash
python -m pytest tests/ -q     # 333 passed（内核 17 + 编排层 316）
```

## 现在能做什么

```bash
python -m dataforge.app.desktop     # 桌面应用（推荐）
python -m dataforge.app.desktop --no-window --port 8000   # 只起服务，用浏览器开

python -m dataforge demo            # 命令行：现场展示缓存与失效传播
python -m dataforge runs            # 列出历史运行
python -m dataforge explain <run>   # 打印血缘
```

桌面应用五个页面：**配置**（字段编辑 / 论文上传 / 变量卡确认 + 门禁阈值 + 实时轨迹）、
**图谱**（真实 DAG，节点按状态着色）、**质量**（门禁明细 + 泄漏检查 + 分布图）、
**数据**（前 200 行预览）、**历史**。

### 论文链路（M3）

```
parse ──> extract ──> confirm ──> schema ──> generate ──┬──> gate ──> export
(规则)     (LLM)      (★人工)      (代码)                 └──> analyze
```

上传 PDF → 模型抽变量卡 → **停下来等人确认** → 确认后续跑。需要设置
`DATAFORGE_API_KEY`（可选 `DATAFORGE_MODEL` / `DATAFORGE_BASE_URL`，默认 DeepSeek）。
**没配密钥时明确报错，不会悄悄退化成规则抽取**——那样抽出来的东西不该被当成论文里的变量。

三条硬规则：

- **相关工作章节永远不送进提示词。** 那是把别人用过的特征误抽成本文变量的主要来源。
  模型自报 `from_related_work` 的变量也默认不勾选。
- **`source_span` 必填。** 给不出论文原文位置的变量基本可以断定是模型编的，直接丢弃。
- **人工确认闸门不可跳过。** 变量抽错了后面全错，而且看不出来——数据照样生成、
  格式正确、质量分很高，只是它描述的不是这篇论文里的现象。

**续跑不是单独实现的「恢复」逻辑，而是指纹缓存的自然结果**：确认结果作为
`confirm` 节点的参数重新提交，上游指纹不变自动命中缓存，只有 `confirm` 及其下游重算。
对论文链路这点很实在——`extract` 是要花钱的那一步。

`demo` 会连跑三次：

1. 全部真算
2. 参数不变 → **五个节点全部命中缓存**
3. 只把行数翻倍 → **`schema` 仍命中缓存**，其余重算

第 3 步就是设计文档 §11 那条验收标准「改一个字段只重跑受影响分支」的现场演示。

## 分层

```
L3  harness/    Gate · Budget · Plateau          ← 唯一允许调 LLM（LLM 尚未接入）
L2  nodes/      桥接内核与外部能力
L1  dataset_generator/   确定性内核，零 LLM 调用（不在本包内）
L0  core/       Artifact · Fingerprint · Trace · Lineage · RunStore
```

依赖**严格单向向下**。`dataset_generator` 不得反向导入本包——这是它 17 个测试能一直跑得动的前提。

## 两条验收标准

| 标准 | 测试 |
|---|---|
| 任意一次运行可完整重放 | `test_replay_reconstructs_entire_run`、`test_full_run_replays_from_trace` |
| 改一个字段只重跑受影响分支 | `test_change_one_param_only_reruns_affected_branch`、`test_changing_num_records_reuses_schema` |

```bash
python -m pytest tests/ -q     # 94 passed（内核 17 + 编排层 77）
```

## 几个容易踩的点

**`code_version` 是必填的。** 指纹 = 上游产物哈希 + 参数 + **代码版本** + seed。
少了代码版本，改了算法指纹不变、缓存命中，你拿到的还是旧算法的数据，**而且不报错**。
`NodeSpec` 在构造时就会拦下空值。

**改了实现忘了升版本号，有兜底。** 执行器会记录每个节点实现的源码哈希，
下次命中缓存时比对，不一致就警告；`Executor(..., strict_drift=True)` 时直接阻断。
CI 里应该开 strict。

**失效按内容传播，不是按位置。** 改了实现但输出逐字节相同 → 下游不必重算；
输出真的变了 → 下游必然重算。前者强行重跑纯属浪费。

**大对象不进图。** 节点之间传的是 `ArtifactRef`（一个哈希），
真正的 DataFrame 落在制品库里。这是不用 LangGraph 的核心原因，见设计文档 §10.1。

**时序任务禁止随机切分。** `split_random` 在 `task_type=time_series_forecast` 时直接报错，
不是警告——随机切分几乎总是错的，而它产生的虚高精度极具迷惑性。

**阈值可以调，但会留痕。** `GateConfig.set()` 把每次放宽都记进 `audit()`，
最终写入数据卡的 `gate_audit` 字段，下游能看到这份数据是在什么标准下产出的。

## 三类下游数据（RL / LLM 训练集 / Agent 测评集）

```bash
python -m dataforge gen rl -n 200            # RL 轨迹
python -m dataforge gen agent-eval -n 50     # Agent 测评集
python -m dataforge gen llm -n 300           # SFT + DPO
```

每类各有**一条不能松的判据**，都做成了会阻断的门禁。

### RL —— 状态转移必须自洽

门禁会**重放环境动力学**，逐条核对 `s'` 是不是真的从 `(s, a)` 走出来的。
这件事从任何统计指标上都看不出来，只能重放。对不上的数据训出来的模型学的是噪声。

环境都是显式动力学（`gridworld` / `inventory`）。库存环境的随机需求作为转移的
一部分记进数据，否则这份数据不可复核。

`epsilon` 别设成 0：**全贪心的轨迹里没有失败样本，离线 RL 学不出「什么不该做」。**

### Agent 测评集 —— 判据必须机器可判且有区分力

每个任务 = 描述 + 工具集 + 初始状态 + **断言**。断言是个受限的小 DSL，
**刻意不支持任意代码**（`python` / `eval` / `shell` 都不在里面）——
测评集要发出去给别人跑，能塞代码就是远程执行漏洞。

门禁做两件事：校验断言定义，以及**用「什么都不做」的空结果试跑一遍**。
在空跑时也能通过的判据区分不出好坏 Agent，直接阻断。

判对错的是代码。LLM-as-Judge 只应用于主观维度的排序。

### LLM 训练集 —— 基准污染零容忍

污染是最危险的一种静默故障：训练集混进评测题目，分数会虚高，
**而虚高的分数看起来像「效果很好」**。

两条踩过的坑：

1. **污染要单独比题面。** 基准题目通常只对应样本的 instruction，
   拿「问题 + 长答案」整段去比，相似度会被答案稀释到接近 0（实测 1.000 → 0.000），
   一条 prompt 逐字照抄评测题的样本反而检不出来。
2. **近重复只报不删。** 实测：合法变体（"中位数" vs "均值"，0.828）
   比真重复（只差一个字，0.760）**相似度还高** ——
   字符级 Jaccard 在原理上分不开这两者。所以只自动删完全重复，
   近重复标出来交给人。这个反直觉的事实写成了测试钉住，防止有人顺手调低阈值。

DPO 的 `rejected` 是把 `chosen` **按明确规则劣化**出来的，劣化方式记进数据。
偏好如果是随机指派的，模型学到的是噪声——而 loss 照样会降。

## 外部源对齐（链接 → 同构合成集）

```bash
python -m dataforge align list                  # 列出已知基准
python -m dataforge align gsm8k -n 500          # 按基准结构生成同构合成集
python -m dataforge align hf://datasets/x/y --network   # 出网取 HF 元数据
```

给一个链接（HF 数据集 / GitHub 仓库 / 已知基准名），取它的**结构**，
生成形状一样、内容是造的同构数据。用途是**冒烟测试、流水线联调、私有基准构造**——
在拿到真实数据前先把训练/评测管线跑通。

### 四条约束

**只取元数据与结构，不下载数据本体。** HF 只碰 `/info` 端点，GitHub 只读仓库元信息
与 schema 文件，都不 clone、不取数据行。取数器有**域名白名单且写死在代码里**——
加一个源必须经过一次显式修改，而不是随便传个 URL 就能出网。

**出网不是默认行为。** `allow_network` 默认关，只查本地基准注册表。
注册表命中就完全不出网，也就不会因为对方接口变动而挂。

**认不出来就说认不出来。** 不猜链接指向什么，也不猜仓库里的数据长什么样。
GitHub 找不到 schema 文件就报错——猜出来的字段会一路流进生成的数据里，
而没人会回头核对。

**映射不了的字段必须回传。** HF 的 `Sequence` / `Image` 这类结构生成器支持不了，
会出现在 `skipped` 里并写进数据卡的 `skipped_fields`。
不报的话用户会以为生成的数据覆盖了全部字段。

### 这里生成的不是原数据集

同构合成数据与原基准的**题目内容无关**。用它刷榜、或在论文里当作真实基准结果报告，
属于学术不端。所以 `synthetic: true` 与来源血缘**强制写进数据卡，不提供关闭选项**。

基准注册表是手工维护的、**必然滞后**，所以它带 `registry_version`；
且只描述结构与评测协议，不含任何原题内容——这一点有测试钉住。

## MCP 服务端（把能力给出去）

```bash
dataforge-mcp                                  # stdio
python -m dataforge.mcp_server --workspace ./ws
```

在 Claude Code / Cursor 的 MCP 配置里指向它，10 个工具即可直接调用：
`df_plan`、`df_generate`、`df_parse_paper`、`df_dataset_from_paper`、
`df_validate`、`df_analyze`、`df_fetch_artifact`、`df_export`、
`df_list_runs`、`df_lineage`。

**红线在这一层要更小心：调用方是另一个 Agent，不会像人那样看一眼再点确认。**
所以闸门必须在协议层就堵住，而不是靠界面：

| 红线 | 怎么堵的 |
|---|---|
| 合成标记不可关闭 | 每次返回都带 notice；`df_export` **强制**同时写数据卡 |
| 阈值不接受外部指定 | `df_generate` 的 schema 里**根本没有** thresholds；硬传也不生效 |
| 人工确认闸门保留 | `df_parse_paper` 只给变量卡；要数据必须再调 `df_dataset_from_paper` 并**显式列出 adopt** |

第三条是重点：外部 Agent **拿不到「一句话直接出数据集」的捷径**。
它必须先看变量卡（含论文原文出处与置信度），再自己决定采纳哪些。
`adopt` 为空直接报错。

另外不回传数据本体——几十万行塞进对话上下文毫无意义。
返回的是产物引用，需要时再用 `df_fetch_artifact` 取预览。

## Planner（自然语言 → 数据集）

界面最上面「说需求」框里写一句话，点「让 Agent 规划」，表单自动填好。

**关键取舍：模型产出的是「计划书」，不是图。**

让模型直接吐 DAG 很有诱惑力，但它能凭空造出不存在的节点类型、连出环、
把输入接到不存在的节点上——校验这些比校验一份扁平的计划书难得多，
出错时也很难给出能喂回模型的错误信息。所以链路是：

```
自然语言 ──► 计划书（LLM，受 PLAN_SPEC 契约约束）
                │
                ├─► autofix（代码）  能确定性修好的直接修
                ├─► check_plan（代码）schema 管不到的语义问题
                └─► plan_to_request（代码，无模型参与）──► 现有流水线
```

**模型碰不到图的结构。** 它填的就是用户本来要手填的那张表单——
所以人工确认是天然的：看一眼、改一改、自己点运行。
Planner 也**不会顺手替你跑一遍**。

`autofix` 只修没有歧义的（min/max 写反、时序任务补 time_column、
target 按 role 唯一纠正）；有歧义的一律报出来交给人。

计划书有语义问题时接口返回 **200 且 `request` 为 null**，而不是 400——
400 只剩一句错误，用户看不到模型到底产出了什么、也就不知道该怎么改需求描述。

同一段需求重复规划会命中缓存，不重复花钱。

## 自动调参（PEVR 外层循环）

界面上勾「不达标时按规则自动调参重试」，或用 `RefineLoop`：

```
Plan → Execute(图) → Verify(门禁) → Refine(调参) → 回到 Execute
```

五个出口，**全部由代码判定**：

| 出口 | 判据 |
|---|---|
| `success` | 门禁全绿且质量分 ≥ 合格线 |
| `plateau` | 连续 N 轮提升 < ε —— 已达当前方法上限 |
| `blocked` | 改参数救不了（泄漏、Schema 非法） |
| `exhausted` | 预算耗尽 |
| `needs_human` | 停在人工确认闸门 |

三条不能松的规则：

**绝不问模型「够好了吗」。** 质量分由 `harness/score.py` 算，而且是**连续**的——
只有通过/不通过的话分数是阶跃的，就分不出「在变好但还不够」和「彻底卡住」。

**Refine 绝不能碰门禁阈值。** 允许它调阈值的话循环一定「收敛」，因为它会把标准降到
自己够得着为止——那不是收敛，是自欺。`sanitize()` 只放行
`num_records` / `missing_rate` / `noise_ratio` 三个白名单参数，
`assert_no_threshold_drift()` 每轮校验。

**停滞检测是必需项不是优化项。** 只写「最多 5 轮」的话，Agent 会在解决不了的问题上
把预算全烧完，而第 2 轮之后一点没进步，最后还给你一份「跑满 5 轮」的努力报告。

调参器先用规则（确定、可测、免费），规则给不出提议时才问模型（`LLMRefiner`），
而模型的提议仍要过 `sanitize()`——**提议归模型，采纳与否归代码**。

诚实说一句：**Refine 只有很少几个杠杆**。泄漏、Schema 非法这类问题改参数修不好，
所以它们会明确停在 `blocked`，而不是把预算白白烧完。

## 前端为什么没上 React

设计文档 §10 选的是 React + TS + Vite，**这一版没照做**：整个前端就是
`web/` 下三个静态文件，没有构建步骤。理由是桌面版要打成 PyInstaller 单文件，
少一条 npm 构建链就少一处会在打包时炸掉的地方；当前 UI 的复杂度也还撑不起
一套构建工具链。等 DAG 编辑、虚拟滚动这类需求上来了再切回去——
后端接口不用动。

## 论文工作流（paperwork，M1）

把 Idea2Paper 的「研究思路 → 可复核论文」九门禁融入为 `dataforge/paperwork/` 子包，
接在数据生成之后，形成完整科研闭环：**论文 → 数据集 → 建模计算 → 手稿 → 机械核验**。

```bash
python -m dataforge.paperwork init <dir>        # 建项目骨架 + 证据模板
python -m dataforge.paperwork demo <dir>        # 教学闭环（newsvendor，可复现）
python -m dataforge.paperwork validate <dir>    # fail-closed 机械检查
python -m dataforge.paperwork run / doi / source / discover / manifest   # 其余子命令
```

九门禁（idea/journal/assets/literature/model/compute/claims/writing/reflection）中，
可代码化的部分（compute/claims/figures/literature/rendering）已落地为脚本与检查；
需要 LLM + 检索的门（idea/journal/model/reflection）保留为 `paperwork/SKILL.md` 指导的
流程，后续按 Planner 哲学（模型产草案、代码校验）接入。

两条哈希链并行不合并：DataForge 的 fingerprint→artifact→lineage，与 paperwork 的
sha256→sources/claims/manifest→validate。交接点：数据卡里的 `synthetic: true` 与来源
延续进 paperwork 的 `sources.json`。设计见 `docs/paperwork-fusion-design.md`。

## 还没做

- **MCP 客户端方向**（接入外部 MCP 服务器）。老实说这条价值不高：
  能接的无非 filesystem / sqlite 这类，而 DataForge 本来就直接读文件。
  真需要时再做。
- **arXiv 源**目前只用于论文链路，没有可映射的数据结构；
  `resolve()` 遇到 arXiv 链接会明确让你走论文链路。
- **GAN / 扩散**：需真实数据 + GPU。当前环境没有 GPU，这条**明确不做**——
  比做一个悄悄降级到统计方法、却在报告里写着用了扩散模型的版本要好。
  真要做时必须先把降级路径和「已降级」的显式告知设计好。
- 内置的 SFT 模板只有三类，样本多样性有限；真用起来应当接模型生成，
  或从自己的语料构造。当前实现验证的是**流程与门禁**，不是语料质量。
