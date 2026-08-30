"""DataForge — 数据集 Agent 的编排层。

分层（依赖严格单向向下，见桌面《DataForge-设计文档》§2.1）::

    L3  harness/   Loop · Gate · Budget · Policy      ← 唯一允许调 LLM
    L2  nodes/     桥接 L1 内核与外部能力
    L1  dataset_generator/   确定性内核（不属于本包，零 LLM 调用）
    L0  core/      Artifact · Fingerprint · Trace · Lineage · RunStore

本包**不得**被 dataset_generator 反向导入。
"""

from .core.artifacts import ArtifactRef, ArtifactStore
from .core.fingerprint import fingerprint, hash_value
from .core.lineage import Lineage
from .core.trace import Trace, TraceEvent
from .graph.dag import DAG
from .graph.executor import ExecStatus, Executor, NodeResult
from .graph.node import NodeSpec

__version__ = "0.1.0"

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "DAG",
    "ExecStatus",
    "Executor",
    "Lineage",
    "NodeResult",
    "NodeSpec",
    "Trace",
    "TraceEvent",
    "fingerprint",
    "hash_value",
]
