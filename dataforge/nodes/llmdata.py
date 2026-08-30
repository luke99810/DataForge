"""LLM 训练集生成：SFT 指令对 / DPO 偏好对 / 多轮对话。

设计文档 §7 列的关键难点是 **去重与基准污染检查**，
因为这两件事都属于静默故障——不报错，只让指标变得好看。

关于 DPO 偏好对还有一条容易被忽略的要求：
**chosen 必须在某个可核查的维度上确实优于 rejected**。
如果偏好是随机指派的，模型学到的是噪声，而 loss 依然会下降。
所以这里的 rejected 是由 chosen **按明确规则劣化**出来的，
劣化方式记进数据，可复核。
"""

from __future__ import annotations

from typing import Any

from ..graph.node import NodeContext
from ..quality.dedup import check_contamination, find_duplicates

V = {
    "generate_sft": "1.0.0",
    "generate_dpo": "1.0.0",
    "llmdata_gate": "1.0.0",
}

# 内置的一小份基准题面样例，用于演示污染检查。
# **不是完整基准**——真用的时候应当从 reference_path 传入自己的评测集。
BUILTIN_BENCHMARK: list[tuple[str, str]] = [
    ("GSM8K-like", "小明有 5 个苹果，又买了 3 个，一共有多少个苹果？"),
    ("GSM8K-like", "一辆车每小时行驶 60 公里，3 小时能行驶多少公里？"),
    ("MMLU-like", "以下哪一项是水的化学式？A. CO2 B. H2O C. NaCl D. O2"),
    ("HumanEval-like", "编写一个函数 reverse_string，接受字符串并返回其反转结果。"),
]

_SFT_TEMPLATES = [
    {
        "topic": "数据处理",
        "instruction": "用 pandas 把 {file} 里 {col} 列的缺失值填成{method}。",
        "output": "可以这样写：\n\n```python\nimport pandas as pd\ndf = pd.read_csv({file!r})\ndf[{col!r}] = df[{col!r}].fillna(df[{col!r}].{fn}())\n```\n\n注意先确认 {col} 是数值列，否则 {fn}() 会报错。",
        "vars": {"file": ["data.csv", "records.csv"], "col": ["age", "score", "price"],
                 "method": ["中位数", "均值"], "fn": ["median", "mean"]},
    },
    {
        "topic": "概念解释",
        "instruction": "什么是{term}？用一两句话说清楚。",
        "output": "{term}指的是{desc}。实际用的时候要注意{caveat}。",
        "vars": {
            "term": ["过拟合", "交叉验证", "正则化"],
            "desc": ["模型在训练集上表现很好、在新数据上明显变差的现象",
                     "把数据分成若干份轮流做验证集来估计泛化能力的做法",
                     "在损失函数里加惩罚项来限制模型复杂度的手段"],
            "caveat": ["训练集指标好不代表可用", "切分方式要和真实场景一致",
                       "惩罚强度需要调，不是越大越好"],
        },
    },
    {
        "topic": "调试",
        "instruction": "运行时报 {error}，可能是什么原因？",
        "output": "{error} 通常有两种原因：{cause1}；{cause2}。先按第一种排查，它更常见。",
        "vars": {
            "error": ["KeyError", "IndexError", "TypeError"],
            "cause1": ["字典里没有这个键", "下标超出了序列长度", "参数类型和函数期望的不一致"],
            "cause2": ["键名拼写或大小写不一致", "循环边界写成了 <= 而不是 <", "某个值意外是 None"],
        },
    },
]

# rejected 的劣化方式。每种都是**可核查的**劣势，不是随机改几个字。
DEGRADE = {
    "truncated": "答案被截断，没说完",
    "vague": "只给了空泛结论，没有可操作的内容",
    "no_caveat": "漏掉了使用前提与注意事项",
    "wrong_detail": "关键细节写错了",
}


def _fill(s: str, m: dict[str, Any]) -> str:
    out = s
    for k, v in m.items():
        out = out.replace("{" + k + "!r}", repr(v)).replace("{" + k + "}", str(v))
    return out


def _sample_pairs(n: int, seed: int) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        tpl = _SFT_TEMPLATES[int(rng.integers(len(_SFT_TEMPLATES)))]
        # 同一模板的不同取值组合，才能拉开样本间的距离
        m = {k: v[int(rng.integers(len(v)))] for k, v in tpl["vars"].items()}
        # 概念解释类的 term/desc/caveat 要对齐，否则会生成自相矛盾的样本
        if tpl["topic"] == "概念解释":
            k = int(rng.integers(len(tpl["vars"]["term"])))
            m = {"term": tpl["vars"]["term"][k], "desc": tpl["vars"]["desc"][k],
                 "caveat": tpl["vars"]["caveat"][k]}
        elif tpl["topic"] == "调试":
            k = int(rng.integers(len(tpl["vars"]["error"])))
            m = {"error": tpl["vars"]["error"][k], "cause1": tpl["vars"]["cause1"][k],
                 "cause2": tpl["vars"]["cause2"][k]}
        rows.append({
            "id": f"sft_{i:05d}",
            "topic": tpl["topic"],
            "instruction": _fill(tpl["instruction"], m),
            "input": "",
            "output": _fill(tpl["output"], m),
        })
    return rows


# ── SFT ───────────────────────────────────────────────────

def generate_sft(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """生成 SFT 指令对，并**在产出前就去重**。

    先生成再去重，而不是让下游自己处理——重复样本流出去之后，
    多半没人会再检查一遍。
    """
    n = int(params.get("num_samples", 200))
    seed = int(params.get("seed", ctx.seed if ctx.seed is not None else 42))
    rows = _sample_pairs(n, seed)

    texts = [r["instruction"] + "\n" + r["output"] for r in rows]
    rep = find_duplicates(texts, threshold=float(params.get("dedup_threshold", 0.9)))

    # 只自动删**完全重复**；近重复只标出不删。
    # 实测（见 quality/dedup.py 顶部）：合法变体（"中位数" vs "均值"，0.828）
    # 比真重复（只差一个字，0.760）还像，阈值在原理上分不开这两者。
    # 自动删近重复会悄悄抹掉数据多样性，而这件事事后无从发现。
    exact_dups = {p.j for p in rep.exact}
    kept = [r for i, r in enumerate(rows) if i not in exact_dups]
    near_only = len(rep.duplicate_indices) - len(exact_dups)

    ctx.note(
        f"生成 {len(rows)} 条，删掉 {len(exact_dups)} 条完全重复后剩 {len(kept)} 条"
        + (f"；另有 {near_only} 条近重复已标出但未删" if near_only else ""),
        generated=len(rows), kept=len(kept), near_flagged=near_only,
    )
    return {
        "format": "sft",
        "samples": kept,
        "dedup": rep.to_dict(),
        "stats": {
            "generated": len(rows),
            "kept": len(kept),
            "exact_removed": len(exact_dups),
            "near_flagged": near_only,
            "duplicate_ratio": round(len(exact_dups) / max(1, len(rows)), 4),
            "by_topic": {t: sum(1 for r in kept if r["topic"] == t)
                         for t in {r["topic"] for r in kept}},
        },
    }


# ── DPO ───────────────────────────────────────────────────

def generate_dpo(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """由 SFT 样本派生 DPO 偏好对。

    rejected 是把 chosen **按明确规则劣化**出来的，劣化方式记进数据。
    偏好如果是随机指派的，模型学到的是噪声——而 loss 照样会降。
    """
    import numpy as np

    sft = inputs.get("sft")
    samples = (sft or {}).get("samples") or []
    if not samples:
        raise ValueError("没有可用的 SFT 样本，无法派生偏好对")

    rng = np.random.default_rng(int(params.get("seed", ctx.seed if ctx.seed is not None else 42)))
    kinds = list(params.get("degrade_kinds") or DEGRADE.keys())
    unknown = [k for k in kinds if k not in DEGRADE]
    if unknown:
        raise ValueError(f"未知的劣化方式 {unknown}，可选：{sorted(DEGRADE)}")

    pairs: list[dict[str, Any]] = []
    for s in samples:
        kind = kinds[int(rng.integers(len(kinds)))]
        good = s["output"]
        if kind == "truncated":
            bad = good[: max(8, len(good) // 3)] + "…"
        elif kind == "vague":
            bad = "这个问题看情况，具体要根据实际需求来定。"
        elif kind == "no_caveat":
            bad = good.split("注意")[0].split("实际用的时候")[0].rstrip("。\n ") or good[:30]
        else:  # wrong_detail
            bad = good.replace("median", "mode").replace("中位数", "众数")
            if bad == good:
                bad = good.replace("不是越大越好", "越大越好")
            if bad == good:
                continue    # 造不出可核查的差异就跳过，不硬凑
        if bad.strip() == good.strip():
            continue
        pairs.append({
            "id": s["id"].replace("sft_", "dpo_"),
            "prompt": s["instruction"],
            "chosen": good,
            "rejected": bad,
            # 记下劣势在哪，才谈得上"可核查"
            "degrade_kind": kind,
            "why_chosen_better": DEGRADE[kind],
        })

    by_kind: dict[str, int] = {}
    for p in pairs:
        by_kind[p["degrade_kind"]] = by_kind.get(p["degrade_kind"], 0) + 1
    ctx.note(f"派生 {len(pairs)} 组偏好对", by_kind=by_kind)
    return {
        "format": "dpo",
        "pairs": pairs,
        "stats": {"num_pairs": len(pairs), "by_degrade_kind": by_kind},
    }


# ── 门禁 ──────────────────────────────────────────────────

def llmdata_gate(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """LLM 训练集门禁：重复率 + **基准污染**。

    污染是这里最该拦的一条：训练集混进评测题目，分数会虚高，
    而虚高的分数看起来像"效果很好"，比报错危险得多。
    """
    from ..graph.executor import NodeBlocked

    data = inputs["dataset"]
    fmt = data.get("format")
    if fmt == "sft":
        items = data.get("samples") or []
        texts = [r["instruction"] + "\n" + r["output"] for r in items]
        # 污染要单独比 **题面**：基准题目通常只对应样本的 instruction，
        # 拿"问题+长答案"整段去比，相似度会被答案稀释到接近 0——
        # 一条 prompt 逐字照抄评测题的样本反而检不出来（实测 1.000 → 0.000）。
        prompts = [r["instruction"] for r in items]
    elif fmt == "dpo":
        items = data.get("pairs") or []
        texts = [r["prompt"] + "\n" + r["chosen"] for r in items]
        prompts = [r["prompt"] for r in items]
    else:
        raise ValueError(f"未知格式 {fmt!r}")

    if not items:
        raise NodeBlocked("[llmdata] 数据集是空的", gate="llmdata", details={"all": []})

    results: list[dict[str, Any]] = []

    # 重复率
    rep = find_duplicates(texts, threshold=float(params.get("dedup_threshold", 0.9)))
    max_dup = float(params.get("max_duplicate_ratio", 0.05))
    results.append({
        "gate": "duplicates", "level": "WARN", "passed": rep.dup_ratio <= max_dup,
        "message": f"重复率 {rep.dup_ratio:.1%}（上限 {max_dup:.0%}）",
        "details": rep.to_dict(),
    })

    # 基准污染 —— 零容忍。题面与整段各查一遍，命中任一即算污染：
    # 「题目泄漏」和「整条样本泄漏」都要抓，漏掉哪种分数都会虚高。
    reference = list(params.get("reference") or BUILTIN_BENCHMARK)
    thr = float(params.get("contamination_threshold", 0.7))
    hits = check_contamination(prompts, reference, threshold=thr)
    seen_idx = {h.index for h in hits}
    hits += [h for h in check_contamination(texts, reference, threshold=thr)
             if h.index not in seen_idx]
    results.append({
        "gate": "contamination", "level": "BLOCK", "passed": not hits,
        "message": ("未检出基准污染" if not hits else
                    f"检出 {len(hits)} 条与评测基准重合的样本 —— "
                    "训练集混进评测题目会让分数虚高，而虚高的分数看起来像「效果很好」"),
        "details": {"hits": [h.to_dict() for h in hits],
                    "reference_size": len(reference)},
    })

    # DPO：chosen 与 rejected 必须真的不同
    if fmt == "dpo":
        same = [p["id"] for p in items if p["chosen"].strip() == p["rejected"].strip()]
        no_reason = [p["id"] for p in items if not p.get("why_chosen_better")]
        results.append({
            "gate": "preference_valid", "level": "BLOCK",
            "passed": not same and not no_reason,
            "message": ("偏好对有效" if not same and not no_reason else
                        f"{len(same)} 组 chosen/rejected 相同、"
                        f"{len(no_reason)} 组说不出优劣依据"),
            "details": {"identical": same[:20], "no_reason": no_reason[:20]},
        })

    blocked = [r for r in results if not r["passed"] and r["level"] == "BLOCK"]
    ctx.note(f"{fmt} 门禁：{len(results)} 项检查，{len(blocked)} 项阻断")
    if blocked:
        raise NodeBlocked(f"[{blocked[0]['gate']}] {blocked[0]['message']}",
                          gate=blocked[0]["gate"], details={"all": results})
    # passed 的口径与主门禁 (harness/gates.GateReport.passed) 保持一致：
    # 只有 BLOCK / ASK 影响放行，WARN 只记录。
    # 否则「重复率略高」这种提醒会把整条流水线判成不通过。
    return {
        "passed": not [r for r in results
                       if not r["passed"] and r["level"] in ("BLOCK", "ASK")],
        "results": results,
        "audit": {"thresholds": {"max_duplicate_ratio": max_dup},
                  "overrides": {}, "relaxed": False},
        "stats": data.get("stats", {}),
    }
