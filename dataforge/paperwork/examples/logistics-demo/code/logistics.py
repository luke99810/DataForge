"""Teaching event-driven simulation: batched vs. no-batch dispatch for on-demand delivery.
Synthetic data (urn:synthetic), fixed seed, standard library only.
Outputs: data/demand.json, results/logistics.json, results/logistics.csv, figures/fig2.svg,
figures/fig3.svg, figures/fig4.svg
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ORIGIN = (0.0, 0.0)
V_KM_PER_MIN = 0.6          # delivery speed
SERVICE_MIN = 3.0           # service time per order (pickup/dropoff)
DEADLINE_OFFSET = 25.0      # deadline = arrival + offset (minutes)
BATCH_CAP = 10              # max orders per batch (S2)
T_MIN = 600.0               # simulation horizon (minutes)


def gen_orders(seed, lam, t_min):
    """Poisson arrivals (rate lam per minute), uniform destinations in [-5,5]^2."""
    rng = random.Random(seed)
    orders = []
    t = 0.0
    while True:
        t += rng.expovariate(lam)
        if t >= t_min:
            break
        orders.append({'id': len(orders), 'arrival': t,
                       'dest_x': rng.uniform(-5.0, 5.0),
                       'dest_y': rng.uniform(-5.0, 5.0),
                       'deadline': t + DEADLINE_OFFSET})
    return orders


def hop(o1, o2):
    return math.hypot(o1['dest_x'] - o2['dest_x'], o1['dest_y'] - o2['dest_y'])


def from_origin(o):
    return math.hypot(o['dest_x'] - ORIGIN[0], o['dest_y'] - ORIGIN[1])


def route_km(batch):
    """Greedy route: origin -> nearest-first chain through destinations -> return."""
    ordered = sorted(batch, key=lambda o: from_origin(o))
    km = from_origin(ordered[0])
    for a, b in zip(ordered, ordered[1:]):
        km += hop(a, b)
    km += from_origin(ordered[-1])          # return to origin
    return km


def simulate(orders, w):
    """Return metrics for a dispatch strategy. w=0 -> no-batch (S1)."""
    n = len(orders)
    if w <= 0:
        batches = [[o] for o in orders]
    else:
        batches = []
        queue = []
        t_open = None
        for o in orders:
            if not queue:
                t_open = o['arrival']
            queue.append(o)
            if len(queue) >= BATCH_CAP or o['arrival'] - t_open >= w:
                batches.append(queue)
                queue = []
        if queue:
            batches.append(queue)
    n_batch = len(batches)
    dist_total = 0.0
    wait_sum = 0.0
    ontime = 0
    for batch in batches:
        depart = max(o['arrival'] for o in batch)
        km = route_km(batch)
        travel = km / V_KM_PER_MIN + SERVICE_MIN * len(batch)
        for o in batch:
            wait_sum += depart - o['arrival']
            if depart + travel <= o['deadline']:
                ontime += 1
        dist_total += km
    return {'n_orders': n, 'n_batch': n_batch, 'dist_total': dist_total,
            'wait_mean': wait_sum / n if n else 0.0, 'ontime': ontime / n if n else 0.0,
            'dist_per_order': dist_total / n if n else 0.0}


def make_chart(xs, series, path, title, ylabel, ymax):
    """Minimal deterministic SVG line chart (two series)."""
    W, H, L, R, T, B = 800, 460, 100, 45, 75, 70
    order = sorted(range(len(xs)), key=xs.__getitem__)
    series = {name: [values[i] for i in order] for name, values in series.items()}
    xs = [xs[i] for i in order]
    plot_w, plot_h = W - L - R, H - T - B
    xmin, xmax = min(xs), max(xs)
    def px(x):
        return L + (x - xmin) / (xmax - xmin) * plot_w if xmax > xmin else L + plot_w / 2
    def py(v):
        return T + (1 - v / ymax) * plot_h
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    lines.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="20" font-size="14" text-anchor="middle" font-family="Times New Roman">{title}</text>')
    lines.append(f'<text x="24" y="{(T+plot_h/2)}" font-size="12" text-anchor="middle" transform="rotate(-90 24 {T+plot_h/2})" font-family="Times New Roman">{ylabel}</text>')
    for gy in range(0, 5):
        v = ymax * gy / 4
        y = py(v)
        lines.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(f'<text x="{L-6}" y="{y+4:.1f}" font-size="11" text-anchor="end" font-family="Times New Roman">{v:.2f}</text>')
    for x in xs:
        lines.append(f'<line x1="{px(x):.1f}" y1="{T}" x2="{px(x):.1f}" y2="{T+plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{px(x):.1f}" y="{T+plot_h+18}" font-size="12" text-anchor="middle" font-family="Times New Roman">{x:g}</text>')
    colors = {'S1': '#0072B2', 'S2': '#009E73', 'S2 ontime %': '#009E73'}
    for index, (name, vals) in enumerate(series.items()):
        pts = ' '.join(f'{px(x):.1f},{py(v):.1f}' for x, v in zip(xs, vals))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors.get(name, "#333333")}" stroke-width="2.5" stroke-dasharray="{"none" if index == 0 else "8 5"}"/>')
        for x, value in zip(xs, vals):
            lines.append(f'<circle cx="{px(x):.1f}" cy="{py(value):.1f}" r="4" fill="{colors.get(name, "#333333")}"/>')
    lx = L + 15
    for name in series:
        lines.append(f'<line x1="{lx}" y1="{T-25}" x2="{lx+20}" y2="{T-25}" stroke="{colors.get(name, "#333333")}" stroke-width="2"/>')
        lines.append(f'<text x="{lx+26}" y="{T-21}" font-size="12" font-family="Times New Roman">{name}</text>')
        lx += 180
    xlabel = 'Batching window (min)' if 'window' in title else 'Arrival rate (orders/min)'
    lines.append(f'<text x="{L + plot_w / 2}" y="{H - 15}" font-size="14" text-anchor="middle" font-family="Times New Roman">{xlabel}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--figdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    seed = int(__import__('os').environ.get('RESEARCH_SEED', str(args.seed)))
    root = Path('.').resolve()

    lam_grid = [0.2, 0.5, 1.0]
    w_grid = [0, 2, 5, 8]

    # Fixed demand draws: one order stream per lambda (shared across strategies).
    demand = {}
    for lam in lam_grid:
        demand[str(lam)] = gen_orders(seed + int(lam * 100), lam, T_MIN)
    (root / 'data' / 'demand.json').write_text(
        json.dumps({'seed': seed, 't_min': T_MIN, 'params': {
            'v_km_per_min': V_KM_PER_MIN, 'service_min': SERVICE_MIN,
            'deadline_offset': DEADLINE_OFFSET, 'batch_cap': BATCH_CAP},
            'demand': demand}, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = []
    for lam in lam_grid:
        orders = demand[str(lam)]
        for strategy, w in (('S1', 0), ('S2', 0), ('S2', 5)):
            m = simulate(orders, w)
            rows.append({'lam': lam, 'strategy': strategy, 'w': w, **m})
        # w sensitivity at lam=0.5 (S2): only w=2 and w=8 (w=0 and w=5 already present)
        if lam == 0.5:
            for w in (2, 8):
                m = simulate(orders, w)
                rows.append({'lam': lam, 'strategy': 'S2', 'w': w, **m})

    # Correctness check: S2 with w=0 must equal S1 (no-batch).
    for lam in lam_grid:
        a = next(r for r in rows if r['lam'] == lam and r['strategy'] == 'S1')
        b = next(r for r in rows if r['lam'] == lam and r['strategy'] == 'S2' and r['w'] == 0)
        if abs(a['dist_total'] - b['dist_total']) > 1e-9 or a['n_batch'] != b['n_batch']:
            raise SystemExit('CORRECTNESS FAIL: S2(w=0) != S1')

    result = {'seed': seed, 'check': {'S2_w0_equals_S1': 'pass'},
              'rows': rows,
              'summary': {
                  'lam_scan': [{'lam': lam, 'S1_n_batch': next(r['n_batch'] for r in rows if r['lam'] == lam and r['strategy'] == 'S1'),
                                'S2_n_batch': next(r['n_batch'] for r in rows if r['lam'] == lam and r['strategy'] == 'S2' and r['w'] == 5),
                                'S1_dist': next(r['dist_total'] for r in rows if r['lam'] == lam and r['strategy'] == 'S1'),
                                'S2_dist': next(r['dist_total'] for r in rows if r['lam'] == lam and r['strategy'] == 'S2' and r['w'] == 5),
                                'S1_wait': next(r['wait_mean'] for r in rows if r['lam'] == lam and r['strategy'] == 'S1'),
                                'S2_wait': next(r['wait_mean'] for r in rows if r['lam'] == lam and r['strategy'] == 'S2' and r['w'] == 5),
                                'S2_ontime': next(r['ontime'] for r in rows if r['lam'] == lam and r['strategy'] == 'S2' and r['w'] == 5)}
                                for lam in lam_grid]}}
    (root / 'results').mkdir(exist_ok=True)
    (root / 'results' / 'logistics.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (root / 'results' / 'logistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figdir = root / args.figdir
    figdir.mkdir(exist_ok=True)
    lam = [r['lam'] for r in result['summary']['lam_scan']]
    s1_batch = [r['S1_n_batch'] for r in result['summary']['lam_scan']]
    s2_batch = [r['S2_n_batch'] for r in result['summary']['lam_scan']]
    s1_wait = [r['S1_wait'] for r in result['summary']['lam_scan']]
    s2_wait = [r['S2_wait'] for r in result['summary']['lam_scan']]
    make_chart(lam, {'S1': s1_batch, 'S2': s2_batch}, figdir / 'fig2.svg',
               'Vehicle batches by arrival rate', 'n_batch', max(s1_batch + s2_batch) * 1.1)
    make_chart(lam, {'S1': s1_wait, 'S2': s2_wait}, figdir / 'fig3.svg',
               'Mean order wait by arrival rate', 'wait_mean (min)', max(s2_wait) * 1.2)
    w = [r['w'] for r in rows if r['lam'] == 0.5 and r['strategy'] == 'S2']
    on = [r['ontime'] * 100 for r in rows if r['lam'] == 0.5 and r['strategy'] == 'S2']
    make_chart(w, {'S2 ontime %': on}, figdir / 'fig4.svg',
               'On-time rate vs batching window (lambda=0.5)', 'ontime %', 100.0)
    print(json.dumps({'status': 'ok', 'n_orders': {str(k): len(v) for k, v in demand.items()},
                      'rows': len(rows)}))


if __name__ == '__main__':
    main()
