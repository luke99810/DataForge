"""DataForge 的论文工作流层（从 Idea2Paper 融入）。

把「研究思路 → 可复核论文」的九门禁流程作为 DataForge 的下游能力：
数据生成（dataforge.paper_pipeline）产出的数据集，在这里继续走
建模计算 → 图表 → 手稿 → 机械核验，形成完整科研闭环。

入口：
    python -m dataforge.paperwork          等价 Idea2Paper 的 scripts/paper.py
    python -m dataforge.paperwork demo <dir>
    python -m dataforge.paperwork validate <dir>

依赖：核心（project/runner/checks/literature）纯标准库；渲染
（render.py 的 docx/LaTeX 版式）需要 python-docx 与 Pillow，见 pyproject 的
``paperwork`` extra。

设计文档见 docs/paperwork-fusion-design.md。
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
