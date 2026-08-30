"""基准注册表。

给一个链接，认出它对应哪个标准测评集，取其**结构与评测协议**，
再生成同构的合成集。

⚠ **这里生成的是同构合成数据，不是原数据集的副本。**
用它刷榜、或在论文里当作真实基准结果报告，属于学术不端。
所以每条记录都带 ``synthetic_notice``，且导出时强制写进数据卡。

用途是**冒烟测试、流水线联调、私有基准构造**——
在拿到真实数据前先把训练/评测管线跑通。

注册表是手工维护的，必然滞后。这一点在
:func:`match` 的返回里显式标了 ``registry_version``，
不认识的链接就老实说不认识，不猜。
"""

from __future__ import annotations

from typing import Any

REGISTRY_VERSION = "2026-08-24"

SYNTHETIC_NOTICE = (
    "本数据集是按公开基准的**结构**生成的合成数据，与原基准的题目内容无关。"
    "不得用于刷榜，也不得作为真实基准评测结果报告。"
)

# 每条记录只描述**结构与协议**，不含任何原题内容。
BENCHMARKS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "name": "GSM8K",
        "task": "小学数学应用题，链式推理后给出数值答案",
        "aliases": ["openai/gsm8k", "gsm8k"],
        "fields": [
            {"name": "question", "type": "string", "description": "题面"},
            {"name": "answer", "type": "string", "description": "含推理过程，末尾 #### 数值"},
        ],
        "splits": {"train": 7473, "test": 1319},
        "protocol": {
            "metric": "exact_match",
            "extract": "取答案里 #### 之后的数值",
            "note": "只比最终数值，不比推理过程",
        },
    },
    "mmlu": {
        "name": "MMLU",
        "task": "多学科四选一单选题",
        "aliases": ["cais/mmlu", "mmlu", "hendrycks_test"],
        "fields": [
            {"name": "question", "type": "string"},
            {"name": "choices", "type": "string", "description": "四个选项，实际是字符串数组"},
            {"name": "answer", "type": "categorical", "categories": ["0", "1", "2", "3"]},
            {"name": "subject", "type": "string"},
        ],
        "splits": {"test": 14042, "validation": 1531, "dev": 285},
        "protocol": {"metric": "accuracy", "note": "通常按 subject 分组后再取宏平均"},
    },
    "humaneval": {
        "name": "HumanEval",
        "task": "由函数签名与文档串补全 Python 实现",
        "aliases": ["openai/openai_humaneval", "humaneval", "openai_humaneval"],
        "fields": [
            {"name": "task_id", "type": "string"},
            {"name": "prompt", "type": "string", "description": "函数签名 + docstring"},
            {"name": "canonical_solution", "type": "string"},
            {"name": "test", "type": "string", "description": "断言测试代码"},
            {"name": "entry_point", "type": "string"},
        ],
        "splits": {"test": 164},
        "protocol": {
            "metric": "pass@k",
            "note": "跑测试判对错——这正是「机器可判」的典型：判据是代码不是模型",
        },
    },
    "swe_bench": {
        "name": "SWE-bench",
        "task": "由 issue 描述产出能通过测试的代码补丁",
        "aliases": ["princeton-nlp/swe-bench", "swe-bench", "swebench"],
        "fields": [
            {"name": "instance_id", "type": "string"},
            {"name": "repo", "type": "string"},
            {"name": "problem_statement", "type": "string"},
            {"name": "patch", "type": "string"},
            {"name": "test_patch", "type": "string"},
            {"name": "base_commit", "type": "string"},
        ],
        "splits": {"test": 2294},
        "protocol": {"metric": "resolved_rate", "note": "应用补丁后跑仓库测试"},
    },
    "imdb": {
        "name": "IMDB",
        "task": "影评二分类情感分析",
        "aliases": ["stanfordnlp/imdb", "imdb"],
        "fields": [
            {"name": "text", "type": "string"},
            {"name": "label", "type": "categorical", "categories": ["neg", "pos"]},
        ],
        "splits": {"train": 25000, "test": 25000},
        "protocol": {"metric": "accuracy"},
    },
}

# 别名 → key
_ALIAS = {a.lower(): k for k, v in BENCHMARKS.items() for a in v["aliases"]}
_ALIAS.update({k: k for k in BENCHMARKS})


def match(identifier: str) -> dict[str, Any] | None:
    """按标识符匹配基准。认不出来返回 None —— **不猜**。"""
    key = _ALIAS.get((identifier or "").strip().lower())
    if key is None:
        # HF 的 org/name 形式，退一步只按 name 匹配
        tail = (identifier or "").rsplit("/", 1)[-1].strip().lower()
        key = _ALIAS.get(tail)
    if key is None:
        return None
    b = BENCHMARKS[key]
    return {
        "key": key,
        "name": b["name"],
        "task": b["task"],
        "fields": [dict(f) for f in b["fields"]],
        "splits": dict(b["splits"]),
        "protocol": dict(b["protocol"]),
        "registry_version": REGISTRY_VERSION,
        "synthetic_notice": SYNTHETIC_NOTICE,
    }


def known() -> list[str]:
    return sorted(BENCHMARKS)
