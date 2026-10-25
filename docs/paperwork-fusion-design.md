# DataForge × Idea2Paper 融合设计

> 目标：把 Idea2Paper 的「论文工作流」作为一层融入 DataForge，形成完整科研闭环：
> **想法 → 文献 → 数据集 → 建模计算 → 论文手稿 → 机械核验**。
> 本文对照两份真实代码撰写（DataForge@`dataforge/`，Idea2Paper@`scripts/`），不是设想稿。

---

## 一、两个项目各自的真实能力

### DataForge（数据侧，主体）

`dataforge/` 五层架构，节点是纯函数 `(inputs, params, ctx) -> value`，`@node(...)` 声明
`NodeSpec`（`node_id` / `kind` / `code_version` / `inputs` / `params`），`DAG` 由
`NodeSpec` 列表组成，执行器负责指纹缓存 + 内容寻址制品 + 血缘 + 轨迹 + 门禁。

现有 pipeline（`dataforge/pipelines.py`）：
`tabular_pipeline` / `timeseries_pipeline` / `paper_pipeline`（论文→数据集）/
`rl_pipeline` / `llm_sft_pipeline` / `agent_eval_pipeline` / `source_aligned_pipeline`。

- `paper_pipeline` 已经是「论文 → 变量卡 → schema → 数据集」，但**只到数据就停**。
- 缺少「数据集 → 建模 → 图表 → 手稿」的下游。

### Idea2Paper（论文侧，被融入方）

`scripts/` 纯标准库 + （渲染时）`python-docx`/`Pillow`，九门禁工作流：
idea → journal → assets → literature → model → compute → claims → writing → reflection。

- `paper.py`（init/demo/validate/discover/source/run/doi/manifest）
- `project.py`（项目目录、来源台账、sha 哈希、scope 限制）
- `runner.py`（执行命令 + 输入输出 SHA + seed + 环境记录）
- `checks.py`（fail-closed 机械检查：run_record/claims/figures/citations/gates/provenance/manifest）
- `literature.py`（Crossref DOI 元数据核验，非全文）
- `render.py`（manuscript.md → LaTeX + Word 双版本）、`svg2png.py`
- 文件式台账：`sources.json` / `claims.json` / `figures.json` / `bibliography.json` / `state.json` / `manifest.json`

---

## 二、融合架构

**物理形态**：DataForge 内新增 `dataforge/paperwork/` 子包，把 Idea2Paper 的脚本与
模板搬入并适配（import 路径、相对引用、依赖）。不动 DataForge 现有五层。

```
L4 界面     app/desktop · __main__ · mcp_server        ── 新增 paperwork CLI 入口
L3 编排     harness · planner                          ── 九门禁映射成 gate 节点（后续）
L2 能力     nodes/…                                    ── 新增 paperwork/（文件式论文工作流）
L1 内核     dataset_generator                          （不动）
L0 来源     core/artifact · fingerprint · lineage      ── 与 paperwork 台账打通（后续）
```

### 搬入清单（Idea2Paper → dataforge/paperwork/）

| Idea2Paper | 搬入后 | 说明 |
|---|---|---|
| `scripts/paper.py` | `paperwork/cli.py` | 子命令入口 |
| `scripts/project.py` | `paperwork/project.py` | 项目原语、来源台账 |
| `scripts/runner.py` | `paperwork/runner.py` | 执行记录 |
| `scripts/checks.py` | `paperwork/checks.py` | 机械检查 |
| `scripts/literature.py` | `paperwork/literature.py` | DOI 核验 |
| `scripts/render.py` | `paperwork/render.py` | LaTeX/Word 渲染 |
| `scripts/svg2png.py` | `paperwork/svg2png.py` | SVG→PNG |
| `scripts/demo.py` | `paperwork/demo.py` | 教学闭环 |
| `scripts/newsvendor.py` | `paperwork/newsvendor.py` | demo 用求解器 |
| `templates/` | `paperwork/templates/` | 证据模板 |
| `references/` | `paperwork/references/` | 方法论文档 |
| `tests/` | `tests/test_paperwork_*.py` | 离线测试 |

### 入口统一

- `python -m dataforge.paperwork` —— 等价 Idea2Paper 的 `python scripts/paper.py`
- 依赖：`python-docx`、`Pillow` 加入 pyproject 的 `paperwork` extra（与 `app` 并列），
  核心依赖零新增（`project/runner/checks/literature` 纯标准库）。

---

## 三、九门禁 → DataForge 的映射

Idea2Paper 的九门禁分两类，融合时区别对待：

**可代码化的门**（搬入后原样保留为文件式流程）：
- compute（`runner.py`）、claims/figures（`checks.py`）、literature（`literature.py`）、
  writing 的渲染（`render.py`）。

**Agent 交互的门**（idea/journal/assets/model/reflection 需要 LLM + 检索）：
- 第一阶段**不强行改造成 DAG 节点**，保留为 SKILL.md 指导的流程；
- 后续用 DataForge 的 planner 哲学（模型产计划书、代码校验）逐步接入。

### 闭环打通点（本次落地）

DataForge 的 `paper_pipeline` 产出数据集（CSV/Parquet + 数据卡）→ 放入 paperwork 的
`data/` 目录 → `runner.py` 跑建模脚本 → `checks.py` 校验 claims/figures → `render.py`
产出 LaTeX/Word 手稿。数据卡里的 `synthetic: true` 与来源血缘延续进 paperwork 的
`sources.json`，避免「合成数据当真实数据写进论文」。

---

## 四、证据链统一（两条哈希链，不强行合并）

两个项目各有一条内容寻址链，语义不同：

- DataForge：`fingerprint`（上游哈希 + 参数 + code_version + seed）→ `artifact` 哈希 → `lineage`。
- Idea2Paper：`sha256(文件)` → `sources.json` / `claims.json` / `manifest.json` → `checks.validate`。

第一阶段**保留两条链并行**，只在交接点对齐：DataForge 导出的数据卡 → paperwork 的
`sources.json` 一行（url=urn、license、sha256）。**不合并**——合并会破坏各自的可复现
语义，且没有用户需求支撑。后续确有需要再谈统一血缘。

---

## 五、实施计划

- **M1（本次）**：搬入 `dataforge/paperwork/`，打通 `python -m dataforge.paperwork`
  与 `python -m dataforge paper`，离线测试跑通，教学 demo 可复现。
- **M2（后续）**：九门禁中 idea/model/reflection 接入 planner 契约（LLM 产草案、代码校验）。
- **M3（后续）**：DataForge 导出 → paperwork 交接的自动桥（数据卡 → sources.json）。
