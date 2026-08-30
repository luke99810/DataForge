"""内容寻址哈希与节点指纹。

设计文档 §5.2 的核心约束：``code_version`` 必须进指纹。
只哈希「输入 + 参数」是个隐蔽的错误——改了生成算法之后指纹没变、缓存命中，
于是拿到的还是旧算法的数据，而且**不报错**。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_LEN = 32  # 截断到 32 hex 字符，够用且日志里好读


def _canonical(obj: Any) -> bytes:
    """把任意可序列化对象转成稳定字节序列。

    要求：同样的语义内容 → 同样的字节。所以 dict 必须排序、float 必须用
    repr 保精度、集合必须排序。
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_default,
    ).encode("utf-8")


def _default(o: Any) -> Any:
    """json 不认识的类型的兜底序列化。"""
    # numpy / pandas 标量
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:  # pragma: no cover - 极少数不可转标量
            pass
    if isinstance(o, (set, frozenset)):
        return sorted(_default(x) if not isinstance(x, (str, int, float, bool)) else x for x in o)
    if isinstance(o, bytes):
        return {"__bytes_sha__": hashlib.sha256(o).hexdigest()}
    if hasattr(o, "to_dict") and callable(o.to_dict):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in sorted(vars(o).items()) if not k.startswith("_")}
    return repr(o)


def hash_value(value: Any) -> str:
    """内容哈希。

    DataFrame 走 pandas 的行哈希 + 列名 + dtype，
    不走 parquet 字节——parquet 会写入创建者版本等元数据，同样的数据
    两次落盘字节可能不同，那样缓存永远命中不了。
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None

    if pd is not None and isinstance(value, pd.DataFrame):
        h = hashlib.sha256()
        # 行内容
        h.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        # 结构：列顺序与类型也是数据的一部分
        h.update(_canonical([list(map(str, value.columns)), [str(d) for d in value.dtypes]]))
        return h.hexdigest()[:HASH_LEN]

    if pd is not None and isinstance(value, pd.Series):
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        h.update(_canonical([str(value.name), str(value.dtype)]))
        return h.hexdigest()[:HASH_LEN]

    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()[:HASH_LEN]

    return hashlib.sha256(_canonical(value)).hexdigest()[:HASH_LEN]


def fingerprint(
    *,
    node_id: str,
    kind: str,
    upstream: dict[str, str],
    params: dict[str, Any],
    code_version: str,
    seed: int | None = None,
) -> str:
    """节点指纹 —— 决定缓存是否命中。

    Args:
        node_id: 节点标识。同一份计算换个节点名不应复用，否则血缘会串。
        kind: 节点类型。
        upstream: {输入名: 上游产物哈希}。上游变了这里就变，失效自动向下传播。
        params: 本节点参数。
        code_version: **必填**。生成算法改了必须同步升版本，否则缓存会喂回旧数据。
        seed: 随机种子。
    """
    if not code_version:
        raise ValueError(
            f"节点 {node_id} 缺少 code_version。"
            "指纹不含代码版本时，改算法不会失效缓存——这是静默返回旧数据的 bug。"
        )
    payload = {
        "node_id": node_id,
        "kind": kind,
        "upstream": dict(sorted(upstream.items())),
        "params": params,
        "code_version": code_version,
        "seed": seed,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:HASH_LEN]


def source_hash(fn: Any) -> str | None:
    """尽力取函数实现的哈希。取不到就返回 None。

    **不进指纹**——源码哈希会因为改注释、改格式而变，进指纹会让缓存
    形同虚设。它只用于开发期的漂移检查：发现「实现变了但 code_version
    没升」，这正是 §5.2 那个静默返回旧数据的故障。

    取不到源码的情况是存在的（exec 出来的函数、内置函数、部分闭包），
    所以调用方必须容忍 None，不能把它当必需品。
    """
    import inspect

    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    # 去掉空白与注释行，避免纯格式改动触发误报
    lines = []
    for line in src.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        lines.append(t)
    if not lines:
        return None
    joined = chr(10).join(lines)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
