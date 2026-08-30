"""DataForge 编排层命令行入口。

    python -m dataforge demo          跑一遍演示流水线，展示缓存与失效传播
    python -m dataforge runs          列出历史运行
    python -m dataforge explain <run> 打印某次运行的血缘

界面（M2）还没做，这个 CLI 是当前唯一的操作入口。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.artifacts import ArtifactStore
from .core.runstore import RunStore
from .graph.executor import Executor
from .pipelines import (
    agent_eval_pipeline, llm_sft_pipeline, rl_pipeline,
    source_aligned_pipeline, tabular_pipeline,
)

DEMO_FIELDS = [
    {"name": "sensor_id", "type": "string", "description": "传感器编号"},
    {"name": "speed", "type": "float", "min": 0, "max": 70, "unit": "mph",
     "origin": "paper", "source_span": {"page": 7, "section": "4.1 Datasets"}},
    {"name": "volume", "type": "integer", "min": 0, "max": 1200, "unit": "veh/5min"},
    {"name": "occupancy", "type": "float", "min": 0, "max": 1},
    {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
]


def _workspace(root: str | Path) -> tuple[ArtifactStore, RunStore, Path]:
    w = Path(root)
    w.mkdir(parents=True, exist_ok=True)
    return ArtifactStore(w / "artifacts"), RunStore(w / "runs.sqlite"), w


def cmd_demo(args: argparse.Namespace) -> int:
    store, rs, w = _workspace(args.workspace)
    ex = Executor(store, rs, workdir=w)

    print("═" * 66)
    print("第一次运行 —— 全部真算")
    print("═" * 66)
    dag = tabular_pipeline(
        fields=DEMO_FIELDS, num_records=args.rows, name="traffic_demo", target="level",
    )
    r1 = ex.run(dag, seed=args.seed, label="demo-1")
    print(r1)
    print(json.dumps(r1.summary(), ensure_ascii=False))

    if not r1.ok:
        print("\n流水线被拦下，符合预期时这也是正确结果。")
        rs.close()
        return 1

    print()
    print("═" * 66)
    print("第二次运行 —— 参数完全一样，应当全部命中缓存")
    print("═" * 66)
    r2 = ex.run(dag, seed=args.seed, label="demo-2")
    print(r2)

    print()
    print("═" * 66)
    print(f"第三次运行 —— 只把行数从 {args.rows} 改成 {args.rows * 2}")
    print("           schema 与它无关，应当仍然命中缓存")
    print("═" * 66)
    dag3 = tabular_pipeline(
        fields=DEMO_FIELDS, num_records=args.rows * 2, name="traffic_demo", target="level",
    )
    r3 = ex.run(dag3, seed=args.seed, label="demo-3")
    print(r3)
    print(f"\n命中缓存：{r3.cached}")
    print(f"重新计算：{r3.computed}")

    card = store.get(r3.results["export"].artifact)
    print()
    print("═" * 66)
    print("数据卡（synthetic 标记不可关闭）")
    print("═" * 66)
    print(json.dumps(
        {k: v for k, v in card.items() if k != "schema"},
        ensure_ascii=False, indent=2,
    ))

    print()
    print("血缘 —— speed 这一列是怎么来的：")
    print(r3.lineage.explain("schema"))
    print(f"\n轨迹：{w / 'traces' / (r3.run_id + '.jsonl')}")
    rs.close()
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    """跑 RL / Agent 测评集 / LLM 训练集三条流水线之一。"""
    store, rs, w = _workspace(args.workspace)
    ex = Executor(store, rs, workdir=w)

    if args.kind == "rl":
        dag = rl_pipeline(env=args.env, num_episodes=args.n)
    elif args.kind == "agent-eval":
        dag = agent_eval_pipeline(num_tasks=args.n)
    else:
        dag = llm_sft_pipeline(num_samples=args.n)

    rep = ex.run(dag, seed=args.seed, label=args.kind)
    print(rep)
    if not rep.ok:
        print(f"\n未通过：{rep.status}")
        rs.close()
        return 1

    gen = rep.results.get("generate") or rep.results.get("sft")
    if gen is not None and gen.artifact is not None:
        data = store.get(gen.artifact)
        print("\n统计：")
        print(json.dumps(data.get("stats", {}), ensure_ascii=False, indent=2))
        if args.kind == "agent-eval" and data.get("tasks"):
            print("\n样例任务（注意每条都带机器可判的断言）：")
            print(json.dumps(data["tasks"][0], ensure_ascii=False, indent=2))
    rs.close()
    return 0


def cmd_evalsuite(args: argparse.Namespace) -> int:
    """测评集：区分度 sweep + 导出 Inspect 格式。"""
    import tempfile

    from .graph.node import NodeContext
    from .nodes.agenteval import generate_agent_eval
    from .nodes.eval_sim import sweep
    from .nodes.inspect_export import export

    ctx = NodeContext(node_id="cli", run_id="cli", seed=args.seed)
    out = generate_agent_eval({}, {"num_tasks": args.n}, ctx)
    tasks = out["tasks"]
    print(f"生成 {len(tasks)} 个任务 · 难度 {out['stats']['by_difficulty']}")
    print(f"类别 {out['stats']['by_category']}")

    if args.export:
        r = export(tasks, args.export, generated_at="cli")
        print()
        print(f"已导出到 {r['dir']}")
        print(f"  {r['num_samples']} 个样本，{r['with_files']} 个带初始文件")
        print(f"  {r['hint']}")
        return 0

    rep = sweep(tasks, Path(tempfile.mkdtemp()))
    print()
    print(f"{'档位':<12}{'总通过率':>9}{'easy':>8}{'medium':>8}{'hard':>8}")
    for prof, r in rep["profiles"].items():
        d = r["by_difficulty"]
        print(f"{prof:<12}{r['pass_rate']:>8.0%} {d.get('easy',0):>7.0%} "
              f"{d.get('medium',0):>7.0%} {d.get('hard',0):>7.0%}")
    print()
    print(f"区分度（最高-最低）：{rep['spread']}  →  "
          f"{'有区分度' if rep['discriminates'] else '★ 无区分度，这套题白出了'}")
    print()
    print(rep["note"])
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    """按外部链接的结构生成同构合成集。"""
    from .sources import HttpFetcher, known

    store, rs, w = _workspace(args.workspace)
    fetcher = HttpFetcher() if args.network else None
    ex = Executor(store, rs, workdir=w, fetcher=fetcher)

    if args.link == "list":
        from .sources import match
        for k in known():
            b = match(k)
            print(f"{k:<12} {b['name']:<12} {b['protocol']['metric']:<16} {b['task']}")
        rs.close()
        return 0

    dag = source_aligned_pipeline(
        link=args.link, num_records=args.n, name=args.name,
        allow_network=args.network,
    )
    rep = ex.run(dag, seed=args.seed, label="align")
    print(rep)
    if not rep.ok:
        rs.close()
        return 1

    card = store.get(rep.results["export"].artifact)
    print()
    print(json.dumps(card, ensure_ascii=False, indent=2))
    rs.close()
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    _, rs, _ = _workspace(args.workspace)
    rows = rs.runs(limit=args.limit)
    if not rows:
        print("还没有运行记录。先跑 `python -m dataforge demo`。")
        return 0
    print(f"{'run_id':<16}{'状态':<12}{'节点':>5}{'命中':>5}  标签")
    for r in rows:
        print(f"{r['run_id']:<16}{r['status']:<12}{r['nodes']:>5}{r['cached']:>5}  {r['label'] or ''}")
    rs.close()
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    _, _, w = _workspace(args.workspace)
    p = w / "lineage" / f"{args.run_id}.json"
    if not p.exists():
        print(f"找不到血缘文件：{p}", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    for nid, e in data.items():
        print(f"\n{nid}  [{e['kind']}]  产物 {e['artifact']}")
        for name, up in sorted(e["upstream"].items()):
            print(f"  ├─ 输入 {name} ← {up}")
        for col, src in sorted(e["columns"].items()):
            print(f"  ├─ 列 {col} ← {json.dumps(src, ensure_ascii=False)}")
        for d in e["decisions"]:
            print(f"  └─ 人工决策：{d['question']} → {d['answer']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataforge", description="DataForge 编排层")
    ap.add_argument("--workspace", default="./.dataforge", help="工作目录（默认 ./.dataforge）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="跑演示流水线，展示缓存与失效传播")
    d.add_argument("--rows", type=int, default=500)
    d.add_argument("--seed", type=int, default=20260823)
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("gen", help="生成 RL / Agent 测评集 / LLM 训练集")
    g.add_argument("kind", choices=["rl", "agent-eval", "llm"])
    g.add_argument("-n", type=int, default=100, help="回合数 / 任务数 / 样本数")
    g.add_argument("--env", default="gridworld", choices=["gridworld", "inventory"])
    g.add_argument("--seed", type=int, default=20260823)
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("align", help="按外部链接的结构生成同构合成集")
    a.add_argument("link", help="基准名 / hf://datasets/x/y / github.com/org/repo，或 list 列出已知基准")
    a.add_argument("-n", type=int, default=500)
    a.add_argument("--name", default="aligned_dataset")
    a.add_argument("--network", action="store_true",
                   help="允许出网取元数据（默认只查本地注册表）")
    a.add_argument("--seed", type=int, default=20260823)
    a.set_defaults(func=cmd_align)

    ev = sub.add_parser("evalsuite", help="测评集区分度 sweep / 导出 Inspect 格式")
    ev.add_argument("-n", type=int, default=40, help="任务数")
    ev.add_argument("--export", metavar="DIR", help="导出成 Inspect 格式到该目录")
    ev.add_argument("--seed", type=int, default=20260824)
    ev.set_defaults(func=cmd_evalsuite)

    r = sub.add_parser("runs", help="列出历史运行")
    r.add_argument("--limit", type=int, default=20)
    r.set_defaults(func=cmd_runs)

    e = sub.add_parser("explain", help="打印某次运行的血缘")
    e.add_argument("run_id")
    e.set_defaults(func=cmd_explain)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
