---
name: Idea2Paper
description: "Use when turning research ideas into reproducible papers."
version: 0.4.0
author: Luke9, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, modeling, evidence, writing]
    related_skills: []
---

# Idea2Paper：从研究思路到可复核论文

适用于运筹、工程、经济、管理及其他需要数据与建模的研究。Agent承担科学推理、合法检索与工具编排；Python只处理重复机械工作。不是自动科学家，不保证创新、真实数据可得性、研究有效性或投稿质量。

## When to Use
- 用户给出思路、资料或初稿，希望完成找资料—选模型—求解—图表—正式写作。
- 不用于补造实验、编造引用、伪造伦理审批或绕过人类投稿批准。

## Prerequisites
- Python 3.10+，仅标准库，无必装框架。用 `terminal` 检查实际解释器。
- 当前包为可搬移本地技能文件，不自动安装到全局 skills。以 `read_file` 加载本文件即可按流程使用。
- 所有路径由调用者传入。执行代码前读源码和运行说明，确认命令及数据权限；shell=False不是沙箱。
- 需要 NumPy/SciPy/优化器等时按模型推荐最小依赖并等待批准；可选 `uv venv <env-dir>` 隔离，不能为了模板先装整个科学栈。

## How to Run
在包目录通过 `terminal(command="python scripts/paper.py --help")` 查看入口。新项目运行 `python scripts/paper.py init <project-dir>`；不覆盖已有目录。教学闭环运行 `python scripts/paper.py demo <new-demo-dir>`，之后 `python scripts/paper.py validate <new-demo-dir>`。具体参数见 README。

## Procedure
1. **问题门**：读用户资料，填写 research_brief.md。区分问题、识别/优化目标、数据、假设、创新边界和可否证标准，并确认目标期刊/文章类型/政策URL。重大模型选择暂停："我推荐X，因为Y。备选：A（权衡），B（权衡）。" 完成标准：用户接受关键选择，未决问题可追溯。
2. **期刊门**：按 `references/journal-adaptation.md` 核查目标期刊官方作者指南（URL、访问日期、当前政策），提取模板/结构/格式/图表/文献/篇幅/投稿材料参数，写入 journal-adaptation.md；再依据 `references/generic-framework.md` 把初稿框架适配为本项目写作框架 framework.md（结构映射、每章要点、图表规划、篇幅分配、格式落地）。完成标准：适配档案与框架可追溯、与期刊要求一致；换刊须重跑本门并复核受影响的 claims/writing 产出。
3. **资料门**：只扫描授权 scope。用 discover 得到不读密码/凭证内容的资产清单；Agent通过 `read_file` 阅读清单内研究资料。缺数据先定位缺字段，再用 `web_search`、`web_extract` 检索公开合规来源，人工确认许可与适用范围；不能将公开可见等同可再分发。完成标准：sources台账、数据字典、口径与许可齐全，缺项明确阻塞。
4. **文献门**：按问题对象、模型机制、数据类型检索类似研究与反例；DOI命令校验元数据。阅读全文支持论断，填写 bibliography.json 的 evidence_read_level、定位和引文。完成标准：逐个论断有支持/反驳记录，研究空白可论证，不能以文献篇数冒充质量。
5. **方法门**：定义变量、单位、目标、约束、识别条件、参数来源、基线、敏感性/稳健性以及验证oracle。科学统计按研究类型适用；理论/确定性优化不硬套P值。重大伦理、数据收集、依赖选型先批准。完成标准：model.md检查表和获批选择写入证据。
6. **计算门**：先写失败测试再实现，执行参数数组命令。run记录退出码、输入输出SHA、环境、种子、超时和日志。失败不通过，保留失败记录；种子必须被模型实际使用。完成标准：可行性/收敛/最优性或统计诊断真实验证，非只看程序退出零。
7. **论断与图表门**：图内文字全英文、Times New Roman实际字体、最终尺寸≥300dpi，图例及所有元素不得相互遮挡或裁切；逐图视觉检查通过后才可交付。claims逐条绑定结果JSON字段和容差；图件绑定产生run与源结果；图表规格遵循 framework.md 的期刊参数。数值含单位、分母、情景与证据等级；核对摘要、表、正文、图完全同口径。完成标准：机械检查通过，Agent另审逻辑支持和外推边界。
8. **写作门**：按 framework.md 章节撰写正式论文（主稿 draft/manuscript.md）。合法阅读全文范本，记录章节功能、段落推进与本研究对应，不复刻原句。按期刊和研究类型组织正式论文，不按脚本写实验日志。正文不写任务执行自述；关键限制不能删。主稿完成后按 `references/rendering.md` 产出 LaTeX 版与 Word 版双版本稿件并出 PDF 预览（LaTeX 优先编译；环境无 LaTeX 时由 Word 版经可用通道导出，保留 .tex 源稿）。完成标准：逐引用语义审计、全文风格映射、真实可得性/作者/伦理/AI声明人工确认；必须完成 references/rendering.md 的逐项版式验收，保存 render-review.json 和逐页预览，只有文件存在或转换退出零不得通过写作门。JCR必须核实年份学科，不能用SJR替代。
9. **反思门**：完成 reviews/reflection.md，检查问题—模型—结果—结论、文献真实性与语义、图表来源、过度因果/创新、负结果、局限，并复查期刊适配仍与目标期刊当前政策一致。修复当前研究后再进入下一项。manifest封存已审工件；validate通过只表示机械完整性，人类对科学判断和投稿负责。

每门在state.json登记pass/blocked/fail、reviewer及证据SHA；未完成不能标pass。不能复制demo的教学自检记录用于真实研究。

## Tool Mapping
- `search_files`/`read_file`：授权本地资产定位与读取；脚本discover提供可复核hash清单。
- `web_search`/`web_extract`：文献、官方数据、目标刊物当前政策；拒绝伪造抓取失败的内容。
- `browser_exec`：合法网页及CDP DOM读取；登录墙需用户，不能绕过付费/权限限制。
- `vision_analyze`：图像/OCR及图表可读性检查，不能由缩略图猜数字。
- `terminal`：测试、计算、CLI；`write_file`/`patch`：记录与脚本；docx/pdf技能只在需要正式格式时加载。

## Pitfalls
- Crossref元数据≠全文阅读≠支持论断；DOI输入原样保留，失败不自动修成另一个编号。
- 引用语法仅支持单项 `[@id]`；lint只检查id存在，不解析任意参考文献风格或判定语义。
- gate是证据存在和hash核验，不验证reviewer身份。manifest无数字签名，不能防恶意同时改写台账和工件。
- discover默认排除常见敏感名字/缓存，不是DLP；非敏感命名也可能藏隐私，扫描范围必须最小化。
- run执行可信命令，日志可能包含命令自身输出的秘密，禁止传密钥参数。超时终止直接子进程，独立派生进程需人工核查。
- 本包设计可跨平台，当前实际验证平台为Windows；未声称已经在Linux/macOS实跑。

## Verification
题名与图表改动必须先用测试复现失败，修复后渲染全部PDF页面并实际视觉审阅。保存每页审阅结果、每图有效dpi和导出通道；XML/hash检查不能代替视觉检查。改图重跑作图与来源链，不为排版擅改模型。
通过 `terminal(command="python -m unittest discover -s tests -v")` 运行离线测试。执行独立新目录demo再validate；对副本篡改结果/图件必须非零退出。真实项目还需逐项人类复核，不能将机械pass写成同行评审。

## References
- [阶段门禁](references/stage-gates.md)
- [期刊适配与写作框架生成](references/journal-adaptation.md)
- [通用论文写作框架](references/generic-framework.md)
- [数据与公开来源](references/data-sources.md)
- [引文语义](references/citation-semantics.md)
- [图表与正式论文](references/figures-writing.md)
- [双版本稿件产出（LaTeX 与 Word）](references/rendering.md)
- [设计来源与政策边界](references/design-sources.md)
- [nature-skills 固定版本审计对照](references/nature-skills-comparison.md)
