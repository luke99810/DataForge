"""去重与基准污染检查。

设计文档 §7 把这两条列为 LLM 训练集的关键难点，理由是它们都属于
**静默故障**：

* 重复样本不会报错，只会让模型在少数几条上过拟合，而 loss 曲线很好看。
* 训练集里混进了评测基准的题目，模型分数会虚高——**这是最危险的一种**，
  因为它看起来像"效果很好"。

两件事都用 **n-gram shingle + Jaccard** 判断。不上 MinHash：
本地数据集规模下暴力比较够快，而 MinHash 会引入近似误差，
在"污染"这种要下结论的场景里，宁可慢一点也要确定。

## 一条必须说清楚的局限

实测（同一批指令数据）：

====================================== =========
文本对                                  相似度
====================================== =========
只差一个字（"保存成" vs "保存为"）        0.760
合法变体（"中位数" vs "均值"）            0.828
不同列（"age" vs "price"）               0.697
完全无关                                 0.000
====================================== =========

**合法变体比真重复还像。** 字符级 Jaccard 在原理上分不开这两者——
任何能抓住 0.760 那条的阈值，必然也会误并 0.828 那条。

所以这里的策略是：

* **完全重复**（归一化后逐字相同）→ 自动删，没有争议
* **近重复** → **只报不删**，交给人看

自动删近重复会悄悄抹掉合法的数据多样性，而这件事事后无从发现。
要真正分开语义重复与合法变体，需要句向量而不是字符重叠——
那是另一个量级的依赖，当前不引入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# 归一化：大小写、空白、标点差异不该让两条重复样本看起来不同。
#
# 标点集合用 re.escape 转义后再拼进字符类，不手写反斜杠——
# 手写时一个 ASCII 引号就能把 raw string 提前截断，
# 后半段变成非 raw 字符串，正则悄悄变成另一个东西。
_PUNCT_CHARS = "".join([
    "　",                 # 全角空格
    "，。！？、；：",
    "“”‘’",
    "（）《》",
    "—",
    ",.!?;:()<>[]{}-_/|",
    "'",
    '"',
    "\\",
])
# 空白与标点必须放进**同一个**字符类，不能写成 `\s+|[punct]+`：
# 那样 "a, b" 里的逗号和空格会被当成两段分别替换，结果留下两个空格，
# 与 "a b" 归一化后不相等——完全重复检测就会漏掉只差标点的样本。
_PUNCT = re.compile(r"[\s" + re.escape(_PUNCT_CHARS) + r"]+")


def normalize(text: str) -> str:
    return _PUNCT.sub(" ", str(text).lower()).strip()


def shingles(text: str, n: int = 5) -> set[str]:
    """字符级 n-gram。

    用字符而不是词：中文没有天然空格，按词切要引入分词器，
    而分词歧义会让"同一句话"算出不同的集合。

    **n 随文本长度自适应。** 短文本用大 n-gram 很脆：
    一句 22 字的话改一个字，5-gram 相似度就掉到 0.57，
    而人一眼就会说这两条是重复的。指令数据大多很短，
    所以短文本降到 3-gram。
    """
    s = normalize(text).replace(" ", "")
    if not s:
        return set()
    n = _adaptive_n(len(s), n)
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _adaptive_n(length: int, requested: int) -> int:
    if length < 40:
        return min(requested, 3)
    if length < 120:
        return min(requested, 4)
    return requested


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass
class DupPair:
    i: int
    j: int
    similarity: float
    kind: str          # exact | near
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"i": self.i, "j": self.j, "similarity": round(self.similarity, 4),
                "kind": self.kind, "preview": self.preview}


@dataclass
class DedupReport:
    total: int = 0
    exact: list[DupPair] = field(default_factory=list)
    near: list[DupPair] = field(default_factory=list)
    duplicate_indices: set[int] = field(default_factory=set)

    @property
    def unique(self) -> int:
        return self.total - len(self.duplicate_indices)

    @property
    def dup_ratio(self) -> float:
        return len(self.duplicate_indices) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "unique": self.unique,
            "duplicate_ratio": round(self.dup_ratio, 4),
            "exact_pairs": [p.to_dict() for p in self.exact[:50]],
            "near_pairs": [p.to_dict() for p in self.near[:50]],
            "exact_count": len(self.exact),
            "near_count": len(self.near),
        }


def find_duplicates(texts: Sequence[str], *, threshold: float = 0.9,
                    n: int = 5, max_pairs: int = 2000) -> DedupReport:
    """找出完全重复与近重复。

    Args:
        threshold: Jaccard 相似度阈值，超过即判为近重复。
        max_pairs: 记录的配对上限，防止极端情况下爆内存。
    """
    rep = DedupReport(total=len(texts))
    norm = [normalize(t) for t in texts]

    # 完全重复：先用哈希桶，O(n)
    seen: dict[str, int] = {}
    for i, t in enumerate(norm):
        if t in seen:
            rep.exact.append(DupPair(seen[t], i, 1.0, "exact", texts[i][:60]))
            rep.duplicate_indices.add(i)
        else:
            seen[t] = i

    # 近重复：只在非完全重复的样本间比
    idx = [i for i in range(len(texts)) if i not in rep.duplicate_indices]
    sh = {i: shingles(texts[i], n) for i in idx}
    for a in range(len(idx)):
        if len(rep.near) >= max_pairs:
            break
        i = idx[a]
        if i in rep.duplicate_indices:
            continue
        for b in range(a + 1, len(idx)):
            j = idx[b]
            if j in rep.duplicate_indices:
                continue
            s = jaccard(sh[i], sh[j])
            if s >= threshold:
                rep.near.append(DupPair(i, j, s, "near", texts[j][:60]))
                rep.duplicate_indices.add(j)
                if len(rep.near) >= max_pairs:
                    break
    return rep


# ── 基准污染 ──────────────────────────────────────────────

@dataclass
class ContaminationHit:
    index: int
    benchmark: str
    similarity: float
    matched: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "benchmark": self.benchmark,
                "similarity": round(self.similarity, 4), "matched": self.matched[:120]}


def check_contamination(
    texts: Sequence[str],
    reference: Iterable[tuple[str, str]],
    *,
    threshold: float = 0.7,
    n: int = 5,
) -> list[ContaminationHit]:
    """检查训练样本是否与评测基准重合。

    Args:
        reference: ``(基准名, 题面)`` 的可迭代对象。
        threshold: 相似度阈值。**刻意比去重阈值低**——
            污染是"宁可错杀不可放过"的场景：漏掉一条，分数就虚高，
            而虚高的分数看起来像"效果很好"，比报错危险得多。
    """
    ref = [(name, shingles(t, n), t) for name, t in reference]
    if not ref:
        return []
    hits: list[ContaminationHit] = []
    for i, t in enumerate(texts):
        st = shingles(t, n)
        if not st:
            continue
        best = max(((jaccard(st, rs), name, raw) for name, rs, raw in ref),
                   key=lambda x: x[0])
        if best[0] >= threshold:
            hits.append(ContaminationHit(i, best[1], best[0], best[2]))
    return hits


def dedup_indices(texts: Sequence[str], *, threshold: float = 0.9) -> list[int]:
    """返回去重后应保留的下标（保留每组的第一条）。"""
    rep = find_duplicates(texts, threshold=threshold)
    return [i for i in range(len(texts)) if i not in rep.duplicate_indices]
