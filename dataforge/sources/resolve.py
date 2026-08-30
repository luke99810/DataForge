"""链接 → 字段定义。

统一入口：不管给的是 HF 数据集、GitHub 仓库还是基准名，
都归结成一份 DataForge 能用的 ``fields``，外加**完整的来源说明**。

解析顺序是刻意的：**先查基准注册表，再去网络**。
注册表命中就不用出网，也就不会因为对方接口变动而挂。
"""

from __future__ import annotations

import json
from typing import Any

from .base import Fetcher, SourceError, SourceRef, identify
from . import hf, registry


def resolve(link: str, fetcher: Fetcher | None = None) -> dict[str, Any]:
    """把一个链接解析成字段定义与来源信息。

    Args:
        fetcher: 为 None 时**只查本地注册表，不出网**。
            这也是默认行为——出网应当是显式选择。
    """
    ref = identify(link)

    # ① 本地注册表优先
    bench = registry.match(ref.id if ref.kind != "unknown" else link)
    if bench is not None:
        return {
            "resolved_by": "registry",
            "ref": ref.to_dict(),
            "benchmark": bench,
            "fields": bench["fields"],
            "skipped": [],
            "splits": bench["splits"],
            "protocol": bench["protocol"],
            "notice": bench["synthetic_notice"],
            "provenance": {
                "source": "builtin_registry",
                "registry_version": bench["registry_version"],
                "note": "只用了基准的结构与评测协议，未接触任何原题内容",
            },
        }

    # ② 压根认不出是什么 —— 先说这个，别拿"注册表里没有"去误导
    if ref.kind == "unknown":
        raise SourceError(
            f"认不出 {link!r} 指向什么。支持 hf://datasets/xxx、"
            "huggingface.co/datasets/xxx、github.com/org/repo，"
            f"或已知基准名（{', '.join(registry.known())}）。"
        )

    # ③ 认出来了但需要出网
    if fetcher is None:
        raise SourceError(
            f"本地注册表里没有 {ref.id!r}（已知：{', '.join(registry.known())}）。"
            "要从网络取元数据，请显式传入 fetcher —— 出网是一个明确的选择，不是默认行为。"
        )

    if ref.kind == "hf_dataset":
        out = hf.fetch_schema(ref.id, fetcher)
        if not out["fields"]:
            raise SourceError(
                f"{ref.id} 的 features 一个都映射不了"
                f"（跳过：{[s['name'] for s in out['skipped']]}）。"
                "结构生成器支持不了这种数据集。"
            )
        return {
            "resolved_by": "huggingface",
            "ref": ref.to_dict(),
            "benchmark": None,
            "fields": out["fields"],
            "skipped": out["skipped"],
            "splits": out["splits"],
            "protocol": None,
            "notice": registry.SYNTHETIC_NOTICE,
            "provenance": {
                "source": "huggingface",
                "dataset_id": ref.id,
                "license": out.get("license"),
                **out["provenance"],
            },
        }

    if ref.kind == "github_repo":
        return _resolve_github(ref, fetcher)

    # arXiv 目前只用于论文链路，没有可映射的数据结构
    raise SourceError(
        f"{ref.kind} 类型的链接（{ref.id}）没有可用于生成数据的结构。"
        "论文请走 df_parse_paper。"
    )


def _resolve_github(ref: SourceRef, fetcher: Fetcher) -> dict[str, Any]:
    """GitHub 仓库：只取仓库元信息，试着找 schema 文件。

    **不 clone、不下载数据文件。** 找到的 schema 文件按 JSON/YAML 解析，
    找不到就老实说找不到——不去猜仓库里的数据长什么样。
    """
    meta = json.loads(fetcher.get(ref.url))
    default_branch = meta.get("default_branch", "main")

    # 只尝试几个约定俗成的位置。找不到不猜。
    candidates = ["schema.json", "dataset_schema.json", "data/schema.json",
                  "schema.yaml", "schema.yml"]
    fields: list[dict[str, Any]] = []
    found: str | None = None
    for c in candidates:
        raw = f"https://raw.githubusercontent.com/{ref.id}/{default_branch}/{c}"
        try:
            text = fetcher.get(raw)
        except SourceError:
            continue
        fields = _parse_schema_text(text, c)
        if fields:
            found = c
            break

    if not fields:
        raise SourceError(
            f"仓库 {ref.id} 里没找到可用的 schema 文件"
            f"（找过：{', '.join(candidates)}）。"
            "不猜数据长什么样——猜出来的字段会一路流进生成的数据里。"
        )

    return {
        "resolved_by": "github",
        "ref": ref.to_dict(),
        "benchmark": None,
        "fields": fields,
        "skipped": [],
        "splits": {},
        "protocol": None,
        "notice": registry.SYNTHETIC_NOTICE,
        "provenance": {
            "source": "github",
            "repo": ref.id,
            "schema_file": found,
            "default_branch": default_branch,
            "stars": meta.get("stargazers_count"),
            "license": (meta.get("license") or {}).get("spdx_id"),
            "note": "只读取了仓库元信息与 schema 文件，未 clone、未下载数据",
        },
    }


_TYPE_ALIAS = {
    "int": "integer", "integer": "integer", "long": "integer",
    "float": "float", "double": "float", "number": "float",
    "str": "string", "string": "string", "text": "string",
    "bool": "boolean", "boolean": "boolean",
    "date": "datetime", "datetime": "datetime", "timestamp": "datetime",
    "category": "categorical", "categorical": "categorical", "enum": "categorical",
}


def _parse_schema_text(text: str, filename: str) -> list[dict[str, Any]]:
    """解析仓库里的 schema 文件。解析不了返回空列表。"""
    doc: Any = None
    if filename.endswith((".yaml", ".yml")):
        try:
            import yaml

            doc = yaml.safe_load(text)
        except Exception:  # noqa: BLE001
            return []
    else:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            return []

    raw_fields = doc.get("fields") if isinstance(doc, dict) else doc
    if not isinstance(raw_fields, list):
        return []

    out: list[dict[str, Any]] = []
    for f in raw_fields:
        if not isinstance(f, dict) or not f.get("name"):
            continue
        t = _TYPE_ALIAS.get(str(f.get("type", "string")).lower())
        if t is None:
            continue
        item: dict[str, Any] = {"name": f["name"], "type": t}
        for k in ("min", "max", "categories", "unit", "description", "start", "end"):
            if f.get(k) is not None:
                item[k] = f[k]
        out.append(item)
    return out
