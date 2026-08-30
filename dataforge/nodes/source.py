"""从外部链接解析结构，用于生成同构合成数据。

设计文档 §6「基准对齐」。§12 的红线在这里体现为三点：

1. **只取元数据与结构**，不下载数据本体
2. 产出**必然**带 ``synthetic`` 声明与来源血缘
3. 映射不了的字段**必须回传**——否则用户会以为生成的数据覆盖了全部字段，
   而少掉的那几列不会有任何提示
"""

from __future__ import annotations

from typing import Any

from ..graph.node import NodeContext
from ..sources.resolve import resolve

V = {"resolve_source": "1.0.0"}


def resolve_source(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """把链接解析成字段定义。

    取数器由 L3 通过 :class:`NodeContext` 注入，**不走 params**——
    params 会进指纹，而 ``HttpFetcher`` 的 repr 里带 token。

    没有注入取数器时**只查本地注册表**：出网是一个显式选择，不是默认行为。
    """
    link = params["link"]
    # allow_network 是个显式开关：即便执行器带着取数器，
    # 节点也只在被明确允许时才出网
    fetcher = ctx.fetcher if params.get("allow_network") else None
    out = resolve(link, fetcher)

    prov = {k: v for k, v in out["provenance"].items() if k != "note"}
    for f in out["fields"]:
        ctx.column_source(f["name"], origin=out["resolved_by"], **prov)

    if out["skipped"]:
        ctx.note(
            f"有 {len(out['skipped'])} 个字段映射不了，已跳过（生成的数据不含这些列）",
            skipped=[s["name"] for s in out["skipped"]],
        )
    bench = (out.get("benchmark") or {}).get("name")
    ctx.note(
        f"由 {out['resolved_by']} 解析出 {len(out['fields'])} 个字段"
        + (f"，匹配到基准 {bench}" if bench else ""),
        link=link,
    )
    return out
