"""HuggingFace 数据集元数据 → DataForge 字段定义。

**只取 features / splits / 数据卡，不下载任何一行数据。**
用的是 datasets-server 的 ``/info`` 接口，它返回结构而不是内容。

映射出来的字段用于生成**同构的合成数据**——形状一样、内容是造的。
所以产出必然带 ``synthetic: true`` 与来源血缘，见 §12 红线一。
"""

from __future__ import annotations

import json
from typing import Any

from .base import Fetcher, SourceError

# HF 的 dtype → DataForge 的 type
_DTYPE = {
    "string": "string", "large_string": "string",
    "int8": "integer", "int16": "integer", "int32": "integer", "int64": "integer",
    "uint8": "integer", "uint16": "integer", "uint32": "integer", "uint64": "integer",
    "float16": "float", "float32": "float", "float64": "float", "double": "float",
    "bool": "boolean",
    "timestamp[s]": "datetime", "timestamp[ms]": "datetime", "date32": "datetime",
}


def map_feature(name: str, feat: Any) -> dict[str, Any] | None:
    """把一个 HF feature 映射成 DataForge 字段。

    映射不了的返回 None —— **宁可少一个字段，也不要编一个类型**。
    编出来的字段会一路流到生成的数据里，而没人会回头核对它对不对。
    """
    if not isinstance(feat, dict):
        return None

    ftype = feat.get("_type")

    if ftype == "Value":
        t = _DTYPE.get(str(feat.get("dtype", "")).lower())
        if t is None:
            return None
        return {"name": name, "type": t, "description": f"HF Value({feat.get('dtype')})"}

    if ftype == "ClassLabel":
        names = feat.get("names")
        if not names:
            return None
        return {"name": name, "type": "categorical",
                "categories": [str(x) for x in names],
                "description": f"HF ClassLabel（{len(names)} 类）"}

    # Sequence / List / Image / Audio 等：结构生成器支持不了，
    # 明确跳过并在报告里说清楚，而不是硬塞成 string
    return None


def parse_info(text: str) -> dict[str, Any]:
    """解析 datasets-server 的 /info 响应。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise SourceError(f"HF 返回的不是合法 JSON：{e}") from e

    info = doc.get("dataset_info")
    if isinstance(info, dict) and "features" not in info:
        # 多配置时是 {config_name: {...}}，取第一个并记下来
        configs = [k for k, v in info.items() if isinstance(v, dict) and "features" in v]
        if configs:
            return {"config": configs[0], **info[configs[0]], "available_configs": configs}
    if isinstance(info, dict):
        return dict(info)
    raise SourceError("HF 响应里没有 dataset_info")


def fetch_schema(dataset_id: str, fetcher: Fetcher) -> dict[str, Any]:
    """取一个 HF 数据集的结构。

    Returns:
        含 ``fields`` / ``skipped`` / ``splits`` / ``provenance`` 的字典。
        ``skipped`` 里是映射不了的 feature —— **必须回传**，
        否则用户会以为生成的数据覆盖了全部字段。
    """
    url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
    info = parse_info(fetcher.get(url))

    features = info.get("features") or {}
    fields: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for name, feat in features.items():
        mapped = map_feature(name, feat)
        if mapped:
            fields.append(mapped)
        else:
            skipped.append({
                "name": name,
                "reason": f"不支持的 feature 类型：{(feat or {}).get('_type', '?')}",
            })

    splits = info.get("splits") or {}
    split_rows = {
        k: (v.get("num_examples") if isinstance(v, dict) else None)
        for k, v in splits.items()
    } if isinstance(splits, dict) else {}

    return {
        "source": "huggingface",
        "dataset_id": dataset_id,
        "config": info.get("config"),
        "available_configs": info.get("available_configs"),
        "fields": fields,
        "skipped": skipped,
        "splits": split_rows,
        "description": (info.get("description") or "")[:600],
        "license": info.get("license"),
        "provenance": {
            "url": url,
            "note": "只读取了 features/splits 元数据，未下载任何数据行",
        },
    }
