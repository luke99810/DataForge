# nature-skills 审计对照

## 审计固定点与许可

- 仓库：`https://github.com/Yuan1z0825/nature-skills.git`
- 固定 commit：`9cecfef6ac683fa59d7d15d2e22f98fa71dacaf`
- 实际读取：`README.md`、`LICENSE`、`skills/nature-data/SKILL.md`、`skills/nature-citation/SKILL.md`、`skills/nature-academic-search/SKILL.md` 及其引用的 core/reference 路由；未执行第三方脚本、未安装依赖、未修改该仓库。
- 许可：仓库 `LICENSE` 为 Apache-2.0。Idea2Paper 为独立实现，仅记录可归因的工作流思想，不复制第三方文件或代码；再分发时保留本项目自身许可证与本审计记录。

## 逐项对照

| 观察到的能力/门禁 | Idea2Paper 实现 | 可验证测试/证据 | 未支持或边界 |
|---|---|---|---|
| 技能采用 router + core/reference 分层 | `SKILL.md` 明确按阶段路由，`references/` 保存数据、引文、图表、stage gates | `references/stage-gates.md` 与 README 入口 | 不复制 nature-skills 的完整内容树；不声称兼容其所有技能名 |
| 文献检索与引用核验分层 | DOI 严格保留、Crossref 成功/失败缓存；bibliography 区分 metadata/abstract/fulltext、claim、quote、semantic review | 既有 `test_crossref_strict_identity_cache_and_failure`；`references/citation-semantics.md` | 不自动判断引文语义、作者归属或全文许可；不提供 nature-academic-search 的多 provider MCP |
| 数据可得性、FAIR、许可与来源 | source ledger 保存 URL、许可、kind、访问时间、SHA；本次新增 provenance 类型约束和观测覆盖范围 | 新增 `test_provenance_requires_explicit_kind_and_coverage`；`checks.provenance` 负例拒绝 | 不自动下载受限/敏感数据，不替作者生成真实仓储 DOI 或伦理批准 |
| 运行产物可追溯 | argv 数组、输入/输出 SHA、seed、环境、stdout/stderr、失败/超时和旧输出归档 | 既有失败、timeout、stale-output、manifest tamper 测试 | run 不是沙箱；GPU/并发确定性仍需模型和人工核验 |
| 阶段门禁与人工责任 | 八阶段固定名单；`publication_ready=true` 现在必须有 `human_approval`、`reviewer_role=human`、批准人和证据 SHA，脚本不能自填通过 | 新增 `test_publication_ready_requires_human_approval_not_self_check`；负例真实失败 | 不验证审阅者身份，不把机械通过变成同行评审或投稿保证 |
| 研究类型适配 | stage-gates 明确统计/经验、理论、优化、仿真的适用项：训练测试分离、多重比较、识别、gap/收敛、基线消融、反例、负结果 | 文档门禁；现有 demo 仅确定性教学枚举 | 尚未实现统计模型/求解器诊断自动解析；须人工填写适用性和证据 |
| 图表与正文同源 | figure 绑定产生 run 的输入/输出，claim 绑定结果字段和容差；图注模板要求单位/情景/证据等级 | 既有 `test_evidence_chain_and_tamper_rejection` | SVG 机械校验不等于视觉审阅；Word/PDF 保真需另加载专门工具 |
| 范本与投稿边界 | `style_reference.md` 强制合法全文、阅读范围、章节映射；README/VERIFICATION 声明不保证创新、期刊或投稿 | demo 明确 publication_ready=false；引用与范本分层文档 | 不提供 Nature 投稿认证、JCR 自动核验或自动批准 |

## 本次独立实现的可验证差异

1. `scripts/checks.py::gates` 对 `publication_ready` 增加 fail-closed 人工批准门；伪造 `agent self-check` 的负例测试失败。
2. 新增 `scripts/checks.py::provenance`，强制受控数据类型；`observed` 缺覆盖范围、`synthetic` 非 `urn:synthetic:`、缺许可/哈希/文件或陈旧工件均拒绝。
3. `templates/state.json` 与 demo state 增加 `human_approval` 结构；更新 `README.md` 与 `references/stage-gates.md` 的实际使用说明。
4. 保留旧测试与断言，先新增测试并实际观察 RED，再实现 GREEN。

## 未支持清单

- 不执行 nature-skills 的安装脚本、MCP server、live provider 或第三方依赖。
- 不把 metadata、摘要或标题相关性当全文语义证据。
- 不自动设计科研模型、替人决定识别假设、创新或期刊目标；不生成付费/敏感数据。
- 不将合成 demo 当实证研究；纯理论项目需要另行证明/反例审计，不能用空 numeric claims 绕过。
- 当前 Windows 实跑；跨平台和真实领域研究仍需项目级验证。
