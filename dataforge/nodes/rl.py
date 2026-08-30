"""RL 轨迹生成。

设计文档 §7 列的难点：**奖励稀疏度、回合长度分布、状态转移自洽**。

其中「状态转移自洽」是唯一能硬检的一条，也是最该检的：
一份 (s, a, r, s', done) 数据如果 s' 根本不是从 (s, a) 走出来的，
拿它训出来的模型学到的是噪声——而这件事从统计指标上完全看不出来。

所以这里的环境都是**显式动力学**：给定 (s, a) 能算出唯一的 s'。
不做「凭空生成看起来像轨迹的数字」——那种数据没有可验证性，
也就没有价值。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph.node import NodeContext

V = {
    "generate_trajectories": "1.0.0",
    "trajectory_gate": "1.0.0",
}


# ── 环境 ──────────────────────────────────────────────────

class GridWorld:
    """N×N 网格，四方向移动，到达目标格得奖励。

    动力学完全确定：撞墙原地不动，否则按方向走一格。
    """

    name = "gridworld"
    actions = ("up", "down", "left", "right")
    _DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def __init__(self, size: int = 5, goal: tuple[int, int] | None = None,
                 step_penalty: float = -0.01) -> None:
        self.size = int(size)
        self.goal = tuple(goal) if goal else (self.size - 1, self.size - 1)
        self.step_penalty = float(step_penalty)

    def spec(self) -> dict[str, Any]:
        return {
            "env": self.name,
            "size": self.size,
            "goal": list(self.goal),
            "actions": list(self.actions),
            "state_fields": ["row", "col"],
            "step_penalty": self.step_penalty,
            "reward_on_goal": 1.0,
        }

    def reset(self, rng: np.random.Generator) -> dict[str, int]:
        while True:
            s = {"row": int(rng.integers(self.size)), "col": int(rng.integers(self.size))}
            if (s["row"], s["col"]) != self.goal:
                return s

    def step(self, s: dict[str, int], a: str) -> tuple[dict[str, int], float, bool]:
        dr, dc = self._DELTA[a]
        r2 = min(max(s["row"] + dr, 0), self.size - 1)
        c2 = min(max(s["col"] + dc, 0), self.size - 1)
        nxt = {"row": r2, "col": c2}
        done = (r2, c2) == self.goal
        return nxt, (1.0 if done else self.step_penalty), done


class Inventory:
    """库存控制：每步决定进货量，随机需求消耗库存。

    动力学含随机需求，所以自洽性检查要按**给定 demand 后**是否成立来判——
    demand 本身作为转移的一部分记进数据里，否则这份数据不可复核。
    """

    name = "inventory"
    actions = ("order_0", "order_5", "order_10", "order_20")
    _ORDER = {"order_0": 0, "order_5": 5, "order_10": 10, "order_20": 20}

    def __init__(self, capacity: int = 60, hold_cost: float = 0.05,
                 stockout_cost: float = 1.0, price: float = 0.4,
                 demand_mean: float = 8.0) -> None:
        self.capacity = int(capacity)
        self.hold_cost = float(hold_cost)
        self.stockout_cost = float(stockout_cost)
        self.price = float(price)
        self.demand_mean = float(demand_mean)

    def spec(self) -> dict[str, Any]:
        return {
            "env": self.name,
            "capacity": self.capacity,
            "actions": list(self.actions),
            "state_fields": ["stock", "day"],
            "hold_cost": self.hold_cost,
            "stockout_cost": self.stockout_cost,
            "price": self.price,
            "demand_mean": self.demand_mean,
        }

    def reset(self, rng: np.random.Generator) -> dict[str, int]:
        return {"stock": int(rng.integers(0, self.capacity // 2)), "day": 0}

    def step(self, s: dict[str, int], a: str, *, demand: int) -> tuple[dict[str, int], float, bool]:
        after_order = min(self.capacity, s["stock"] + self._ORDER[a])
        sold = min(after_order, demand)
        missed = demand - sold
        stock2 = after_order - sold
        reward = (
            self.price * sold
            - self.hold_cost * stock2
            - self.stockout_cost * missed
        )
        nxt = {"stock": int(stock2), "day": s["day"] + 1}
        return nxt, float(reward), False


ENVS = {"gridworld": GridWorld, "inventory": Inventory}


# ── 策略 ──────────────────────────────────────────────────

def _policy_action(env: Any, s: dict[str, int], rng: np.random.Generator,
                   epsilon: float) -> str:
    """ε-贪心：以 1-ε 概率走一步"像样"的动作，否则随机。

    不是最优策略，也**不该是**——全最优的轨迹里没有失败样本，
    离线 RL 学不出「什么不该做」。ε 控制数据的多样性。
    """
    if rng.random() < epsilon:
        return str(rng.choice(env.actions))

    if isinstance(env, GridWorld):
        gr, gc = env.goal
        if s["row"] != gr:
            return "down" if gr > s["row"] else "up"
        return "right" if gc > s["col"] else "left"

    # 库存：低于安全线就多进货
    stock = s["stock"]
    if stock < env.demand_mean:
        return "order_20"
    if stock < env.demand_mean * 2:
        return "order_10"
    return "order_0"


# ── 生成 ──────────────────────────────────────────────────

def generate_trajectories(inputs: dict[str, Any], params: dict[str, Any],
                          ctx: NodeContext) -> dict:
    """生成 (s, a, r, s', done) 轨迹。

    产出是 dict（而不是 DataFrame），因为轨迹是**嵌套结构**：
    回合边界一旦拍平进表格，就再也检不出「done 之后还有转移」这类错误。
    """
    env_name = params.get("env", "gridworld")
    cls = ENVS.get(env_name)
    if cls is None:
        raise ValueError(f"未知环境 {env_name!r}，可选：{sorted(ENVS)}")
    env = cls(**(params.get("env_params") or {}))

    n_ep = int(params.get("num_episodes", 200))
    max_steps = int(params.get("max_steps", 50))
    epsilon = float(params.get("epsilon", 0.2))
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon 必须在 [0,1]，得到 {epsilon}")
    seed = int(params.get("seed", ctx.seed if ctx.seed is not None else 42))
    rng = np.random.default_rng(seed)

    episodes: list[dict[str, Any]] = []
    for i in range(n_ep):
        s = env.reset(rng)
        steps: list[dict[str, Any]] = []
        total = 0.0
        for t in range(max_steps):
            a = _policy_action(env, s, rng, epsilon)
            if isinstance(env, Inventory):
                demand = int(rng.poisson(env.demand_mean))
                s2, r, done = env.step(s, a, demand=demand)
                extra = {"demand": demand}
            else:
                s2, r, done = env.step(s, a)
                extra = {}
            steps.append({
                "t": t, "state": s, "action": a, "reward": round(float(r), 6),
                "next_state": s2, "done": bool(done), **extra,
            })
            total += r
            s = s2
            if done:
                break
        episodes.append({
            "episode_id": i,
            "steps": steps,
            "length": len(steps),
            "return": round(total, 6),
            "terminated": bool(steps[-1]["done"]) if steps else False,
        })

    lengths = [e["length"] for e in episodes]
    returns = [e["return"] for e in episodes]
    n_steps = sum(lengths)
    nonzero = sum(
        1 for e in episodes for st in e["steps"]
        if abs(st["reward"]) > 1e-9 and st["reward"] > 0
    )
    ctx.note(
        f"{env_name}: {n_ep} 回合 / {n_steps} 步，终止率 "
        f"{sum(1 for e in episodes if e['terminated']) / max(1, n_ep):.0%}",
        episodes=n_ep, steps=n_steps,
    )
    return {
        "env_spec": env.spec(),
        "episodes": episodes,
        "stats": {
            "num_episodes": n_ep,
            "num_steps": n_steps,
            "mean_length": round(float(np.mean(lengths)), 3) if lengths else 0.0,
            "mean_return": round(float(np.mean(returns)), 4) if returns else 0.0,
            "terminated_ratio": round(
                sum(1 for e in episodes if e["terminated"]) / max(1, n_ep), 4),
            # 正奖励步占比：太低说明奖励过于稀疏，离线 RL 基本学不动
            "positive_reward_ratio": round(nonzero / max(1, n_steps), 4),
        },
        "seed": seed,
        "epsilon": epsilon,
    }


# ── 校验 ──────────────────────────────────────────────────

def check_trajectories(data: dict[str, Any], *, sample: int | None = None) -> list[dict[str, Any]]:
    """轨迹自洽性检查。

    最重要的一条：**重放动力学，核对 s' 是不是真的从 (s, a) 走出来的**。
    这件事从任何统计指标上都看不出来，只能逐条重放。
    """
    findings: list[dict[str, Any]] = []
    spec = data.get("env_spec") or {}
    cls = ENVS.get(spec.get("env", ""))
    episodes = data.get("episodes") or []
    if not episodes:
        return [{"severity": "error", "kind": "empty", "message": "没有任何回合"}]

    env = None
    if cls is not None:
        kwargs = {k: v for k, v in spec.items()
                  if k in cls.__init__.__code__.co_varnames and k != "self"}
        if spec.get("env") == "gridworld" and "goal" in spec:
            kwargs["goal"] = tuple(spec["goal"])
        env = cls(**kwargs)

    rng_ep = episodes if sample is None else episodes[:sample]
    bad_transitions = 0
    for e in rng_ep:
        steps = e.get("steps") or []
        if not steps:
            findings.append({"severity": "error", "kind": "empty_episode",
                             "message": f"回合 {e.get('episode_id')} 没有任何步"})
            continue

        # done 只能出现在最后一步
        for st in steps[:-1]:
            if st.get("done"):
                findings.append({
                    "severity": "error", "kind": "done_not_last",
                    "message": f"回合 {e.get('episode_id')} 第 {st['t']} 步已 done，后面却还有转移",
                })
                break

        # t 必须连续
        ts = [st.get("t") for st in steps]
        if ts != list(range(len(steps))):
            findings.append({"severity": "error", "kind": "bad_time_index",
                             "message": f"回合 {e.get('episode_id')} 的时间索引不连续"})

        # 相邻步的 next_state 必须等于下一步的 state
        for a, b in zip(steps, steps[1:]):
            if a.get("next_state") != b.get("state"):
                findings.append({
                    "severity": "error", "kind": "state_chain_broken",
                    "message": (f"回合 {e.get('episode_id')} 第 {a['t']} 步的 next_state "
                                f"与第 {b['t']} 步的 state 不一致"),
                })
                break

        # ★ 重放动力学
        if env is not None:
            for st in steps:
                try:
                    if isinstance(env, Inventory):
                        s2, r, done = env.step(st["state"], st["action"],
                                               demand=int(st.get("demand", 0)))
                    else:
                        s2, r, done = env.step(st["state"], st["action"])
                except Exception:  # noqa: BLE001 - 动作非法等
                    bad_transitions += 1
                    continue
                if s2 != st.get("next_state") or abs(r - float(st.get("reward", 0))) > 1e-6:
                    bad_transitions += 1

    if bad_transitions:
        findings.append({
            "severity": "error", "kind": "dynamics_mismatch",
            "message": (f"{bad_transitions} 条转移重放不上：s' 或 reward 与环境动力学不符。"
                        "这类数据训出来的模型学的是噪声，而统计指标上完全看不出来。"),
            "count": bad_transitions,
        })

    stats = data.get("stats") or {}
    pr = float(stats.get("positive_reward_ratio", 0))
    if pr < 0.01:
        findings.append({
            "severity": "warning", "kind": "sparse_reward",
            "message": f"正奖励步只占 {pr:.2%}，奖励过于稀疏，离线 RL 可能学不动",
        })
    if float(stats.get("terminated_ratio", 1)) < 0.1 and spec.get("env") == "gridworld":
        findings.append({
            "severity": "warning", "kind": "rarely_terminates",
            "message": "绝大多数回合没能到达目标，轨迹里几乎没有成功样本",
        })
    return findings


def trajectory_gate(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """轨迹质量门禁 —— 转移不自洽直接阻断。"""
    from ..graph.executor import NodeBlocked

    data = inputs["trajectories"]
    findings = check_trajectories(data, sample=params.get("sample"))
    errors = [f for f in findings if f["severity"] == "error"]
    warnings = [f for f in findings if f["severity"] == "warning"]

    ctx.note(f"轨迹检查：{len(errors)} 个 error / {len(warnings)} 个 warning")
    if errors:
        raise NodeBlocked(
            f"[trajectory] {errors[0]['message']}",
            gate="trajectory",
            details={"all": [
                {"gate": "trajectory", "level": "BLOCK", "passed": False,
                 "message": errors[0]["message"], "details": {"findings": findings}},
            ]},
        )
    return {
        "passed": True,
        "results": [{"gate": "trajectory", "level": "BLOCK", "passed": True,
                     "message": f"转移自洽，{len(warnings)} 个提醒",
                     "details": {"findings": findings}}],
        "audit": {"thresholds": {}, "overrides": {}, "relaxed": False},
        "stats": data.get("stats", {}),
    }
