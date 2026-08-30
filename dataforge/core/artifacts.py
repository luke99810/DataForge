"""内容寻址的制品库。

为什么大对象不能放进图的状态里（设计文档 §10.1）：
节点产出的是几百 MB 的数组和 parquet，塞进 state 每一步都要序列化一遍，
缓存的开销会超过重算。所以图里只传 :class:`ArtifactRef`（一个哈希），
真正的数据落在这里。
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fingerprint import hash_value


@dataclass(frozen=True)
class ArtifactRef:
    """指向制品库里一份数据的引用。图中流动的是它，不是数据本身。"""

    hash: str
    kind: str  # dataframe | json | text | bytes | pickle
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"hash": self.hash, "kind": self.kind, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArtifactRef":
        return cls(hash=d["hash"], kind=d["kind"], meta=d.get("meta", {}))

    def __repr__(self) -> str:  # 日志里好读
        return f"<{self.kind}:{self.hash[:10]}>"


_EXT = {
    "dataframe": ".parquet",
    "json": ".json",
    "text": ".txt",
    "bytes": ".bin",
    "pickle": ".pkl",
}


class ArtifactStore:
    """按内容哈希存取制品。

    同样的内容只存一份——不同运行产出相同数据时天然去重。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── 内部 ──────────────────────────────────────────────
    def _path(self, h: str, kind: str) -> Path:
        # 两级分桶，避免单目录几万个文件
        d = self.root / h[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{h}{_EXT.get(kind, '.bin')}"

    @staticmethod
    def _kind_of(value: Any) -> str:
        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                return "dataframe"
        except ImportError:  # pragma: no cover
            pass
        if isinstance(value, bytes):
            return "bytes"
        if isinstance(value, str):
            return "text"
        if isinstance(value, (dict, list, int, float, bool)) or value is None:
            return "json"
        return "pickle"

    # ── 对外 ──────────────────────────────────────────────
    def put(self, value: Any, *, kind: str | None = None, meta: dict[str, Any] | None = None) -> ArtifactRef:
        """存入一份数据，返回引用。已存在则直接复用，不重复写盘。"""
        kind = kind or self._kind_of(value)
        h = hash_value(value)
        p = self._path(h, kind)

        if not p.exists():
            tmp = p.with_suffix(p.suffix + ".tmp")
            if kind == "dataframe":
                value.to_parquet(tmp, index=True)
            elif kind == "json":
                tmp.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            elif kind == "text":
                tmp.write_text(value, encoding="utf-8")
            elif kind == "bytes":
                tmp.write_bytes(value)
            else:
                tmp.write_bytes(pickle.dumps(value, protocol=4))
            # 先写临时文件再改名：进程中途被杀不会留下半截制品被后续误当命中
            tmp.replace(p)

        auto_meta = self._describe(value, kind)
        auto_meta.update(meta or {})
        return ArtifactRef(hash=h, kind=kind, meta=auto_meta)

    def get(self, ref: ArtifactRef) -> Any:
        """按引用取回数据。"""
        p = self._path(ref.hash, ref.kind)
        if not p.exists():
            raise FileNotFoundError(f"制品不存在：{ref}（路径 {p}）")
        if ref.kind == "dataframe":
            import pandas as pd

            return pd.read_parquet(p)
        if ref.kind == "json":
            return json.loads(p.read_text(encoding="utf-8"))
        if ref.kind == "text":
            return p.read_text(encoding="utf-8")
        if ref.kind == "bytes":
            return p.read_bytes()
        return pickle.loads(p.read_bytes())

    def exists(self, ref: ArtifactRef) -> bool:
        return self._path(ref.hash, ref.kind).exists()

    def path_of(self, ref: ArtifactRef) -> Path:
        return self._path(ref.hash, ref.kind)

    def clear(self) -> None:
        """清空制品库。仅供测试与显式的「清理缓存」操作使用。"""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _describe(value: Any, kind: str) -> dict[str, Any]:
        """记录一点结构信息，让界面不必加载整份数据就能显示概览。"""
        if kind == "dataframe":
            return {
                "rows": int(len(value)),
                "cols": int(len(value.columns)),
                "columns": [str(c) for c in value.columns],
            }
        if kind in ("json", "text"):
            try:
                return {"size": len(value)}
            except TypeError:
                return {}
        if kind == "bytes":
            return {"size": len(value)}
        return {}
