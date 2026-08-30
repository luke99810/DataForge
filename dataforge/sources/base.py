"""外部数据源的取数抽象。

设计文档 §12 红线二：**只读取元数据与结构，不下载数据本体，不缓存，不再分发。**
带许可限制的数据集只取其 schema 形状。

这一层做成抽象还有个现实原因：**测试不该依赖网络**。
网络会让测试在别人机器上、在 CI 里、在断网时随机变红，
而变红的原因和被测逻辑毫无关系。所以取数走 :class:`Fetcher`，
测试注入 :class:`StubFetcher`。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# 只允许访问这几个域名。写死是刻意的：
# 让"顺手加个别的源"必须经过一次显式修改，而不是随便传个 URL 就能出网。
ALLOWED_HOSTS = (
    "datasets-server.huggingface.co",
    "huggingface.co",
    "api.github.com",
    "raw.githubusercontent.com",
    "export.arxiv.org",
    "arxiv.org",
)


class SourceError(RuntimeError):
    """取数失败。"""


class Fetcher(Protocol):
    """取一个 URL 的文本内容。"""

    def get(self, url: str, *, timeout: float = 15.0) -> str: ...


@dataclass
class HttpFetcher:
    """真实网络取数。只允许白名单域名。"""

    user_agent: str = "DataForge/0.1 (metadata-only)"
    token: str | None = None

    def get(self, url: str, *, timeout: float = 15.0) -> str:
        import urllib.parse
        import urllib.request

        host = urllib.parse.urlparse(url).netloc.lower()
        if host not in ALLOWED_HOSTS:
            raise SourceError(
                f"域名 {host!r} 不在白名单里。只读取元数据的源需要显式加进 ALLOWED_HOSTS。"
            )
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - 网络错误要变成可读信息
            raise SourceError(f"取 {url} 失败：{type(e).__name__}: {e}") from e


@dataclass
class StubFetcher:
    """测试与离线演示用。按 URL 子串匹配预设响应。"""

    responses: dict[str, Any] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def get(self, url: str, *, timeout: float = 15.0) -> str:
        self.calls.append(url)
        for needle, payload in self.responses.items():
            if needle in url:
                return payload if isinstance(payload, str) else json.dumps(payload)
        raise SourceError(f"StubFetcher 没有匹配 {url} 的预设响应")


# ── 链接识别 ──────────────────────────────────────────────

@dataclass
class SourceRef:
    """一个被识别出来的外部引用。"""

    kind: str                    # hf_dataset | github_repo | arxiv | unknown
    id: str
    url: str
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id, "url": self.url, "raw": self.raw}


_PATTERNS = [
    ("hf_dataset", re.compile(r"^hf://datasets/([\w\-./]+)$", re.I)),
    ("hf_dataset", re.compile(r"huggingface\.co/datasets/([\w\-]+/[\w\-.]+)", re.I)),
    ("hf_dataset", re.compile(r"huggingface\.co/datasets/([\w\-.]+)$", re.I)),
    ("github_repo", re.compile(r"github\.com/([\w\-.]+/[\w\-.]+?)(?:\.git)?(?:/|$)", re.I)),
    ("arxiv", re.compile(r"arxiv\.org/abs/([\d.]+v?\d*)", re.I)),
    ("arxiv", re.compile(r"^arxiv:([\d.]+v?\d*)$", re.I)),
]


def identify(link: str) -> SourceRef:
    """认出链接指向什么。认不出来就说认不出来，不猜。"""
    s = (link or "").strip()
    if not s:
        raise SourceError("链接是空的")
    for kind, pat in _PATTERNS:
        m = pat.search(s)
        if m:
            ident = m.group(1)
            if kind == "hf_dataset":
                url = f"https://datasets-server.huggingface.co/info?dataset={ident}"
            elif kind == "github_repo":
                url = f"https://api.github.com/repos/{ident}"
            else:
                url = f"http://export.arxiv.org/api/query?id_list={ident}"
            return SourceRef(kind=kind, id=ident, url=url, raw=s)
    return SourceRef(kind="unknown", id=s, url="", raw=s)


# ── 带缓存的取数 ──────────────────────────────────────────

class CachedFetcher:
    """给任意 Fetcher 加一层内存缓存。

    只缓存**元数据**，且只在进程内。不落盘——落盘就变成了
    "再分发别人的数据"，哪怕只是元数据也应当谨慎。
    """

    def __init__(self, inner: Fetcher, ttl: float = 900.0) -> None:
        self.inner = inner
        self.ttl = ttl
        self._cache: dict[str, tuple[float, str]] = {}

    def get(self, url: str, *, timeout: float = 15.0) -> str:
        now = time.time()
        hit = self._cache.get(url)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        text = self.inner.get(url, timeout=timeout)
        self._cache[url] = (now, text)
        return text
