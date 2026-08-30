"""节点定义。

节点是一个纯函数：``(inputs, params, ctx) -> value``。

它拿到的 ``inputs`` 是**已经取回的数据**（不是引用），产出的 ``value``
由执行器负责存进制品库。节点自己不碰制品库，也不知道缓存的存在——
这样节点才好单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# 节点类型，对应设计文档 §5.1
NODE_KINDS = (
    "SOURCE",     # 论文 / 链接 / 本地文件
    "EXTRACT",    # 抽取变量卡
    "SCHEMA",     # 字段定义
    "GENERATE",   # 生成数据
    "TRANSFORM",  # 变换
    "GATE",       # 质量门禁 —— 唯一可阻断的类型
    "ANALYZE",    # 统计与可视化
    "EXPORT",     # 导出
)


class NodeFn(Protocol):
    def __call__(self, inputs: dict[str, Any], params: dict[str, Any], ctx: "NodeContext") -> Any: ...


@dataclass
class NodeContext:
    """执行期传给节点的上下文。"""

    node_id: str
    run_id: str
    seed: int | None = None
    workdir: Any = None
    # 由 L3 执行器注入。节点不知道模型怎么配的，所以仍可脱离模型单测
    # （设计文档 §2.1：只有 L3 允许构造 LLM 客户端）
    llm: Any = None
    # 外部取数器。**必须走这里而不是 params**：params 会进指纹，
    # 而 HttpFetcher 的 repr 里带 token —— 那会把密钥写进指纹与轨迹。
    fetcher: Any = None
    # 节点可以往这里写想进血缘的东西（列来源、人工决策）
    lineage_notes: dict[str, Any] = field(default_factory=dict)
    # 节点可以往这里写想进轨迹的东西
    trace_notes: list[dict[str, Any]] = field(default_factory=list)

    def note(self, msg: str, **data: Any) -> None:
        self.trace_notes.append({"msg": msg, **data})

    def column_source(self, column: str, **source: Any) -> None:
        self.lineage_notes.setdefault("columns", {})[column] = source

    def decision(self, question: str, answer: Any, by: str = "human", note: str = "") -> None:
        self.lineage_notes.setdefault("decisions", []).append(
            {"question": question, "answer": answer, "by": by, "note": note}
        )


@dataclass
class NodeSpec:
    """一个节点的完整声明。

    Attributes:
        node_id: 图内唯一。
        kind: NODE_KINDS 之一。
        fn: 计算函数。
        inputs: {输入名: 上游 node_id}。
        params: 参数，进指纹。
        code_version: **必填**，算法改了要升。见 fingerprint.fingerprint 的说明。
        cacheable: 置 False 则每次都重算（例如带外部时间的源节点）。
        artifact_kind: 强制指定产物存储类型，None 则自动判断。
    """

    node_id: str
    kind: str
    fn: NodeFn
    inputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    code_version: str = ""
    cacheable: bool = True
    artifact_kind: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"未知节点类型 {self.kind!r}，可选：{NODE_KINDS}")
        if not self.code_version:
            raise ValueError(
                f"节点 {self.node_id} 必须声明 code_version —— 见设计文档 §5.2，"
                "缺了它改算法不会失效缓存。"
            )


def node(
    node_id: str,
    kind: str,
    *,
    inputs: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    code_version: str,
    cacheable: bool = True,
    artifact_kind: str | None = None,
    description: str = "",
) -> Callable[[NodeFn], NodeSpec]:
    """把一个函数声明成节点的装饰器。"""

    def deco(fn: NodeFn) -> NodeSpec:
        return NodeSpec(
            node_id=node_id,
            kind=kind,
            fn=fn,
            inputs=dict(inputs or {}),
            params=dict(params or {}),
            code_version=code_version,
            cacheable=cacheable,
            artifact_kind=artifact_kind,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
        )

    return deco
