"""Teaching-only finite-demand enumeration; no empirical claims."""
import csv
import json
from pathlib import Path


def solve(data):
    rows = []
    for quantity in range(data['max_order'] + 1):
        profit = sum(prob * (data['price'] * min(quantity, demand) - data['cost'] * quantity)
                     for demand, prob in zip(data['demand'], data['probability']))
        rows.append({'quantity': quantity, 'expected_profit': profit})
    best = max(rows, key=lambda r: r['expected_profit'])
    return rows, {'optimal_q': best['quantity'], 'expected_profit': best['expected_profit'],
                  'synthetic': True, 'method': 'exhaustive finite enumeration'}


def draw(rows, path):
    points = ' '.join(f"{60 + r['quantity'] * 85},{240 - r['expected_profit'] * 25}" for r in rows)
    ticks = ''.join(f'<text x="{60 + r["quantity"] * 85}" y="265">{r["quantity"]}</text>' for r in rows)
    yticks = ''.join(f'<text x="25" y="{244 - value * 25}">{value}</text>' for value in range(0, 7, 2))
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="500" height="330" viewBox="0 0 500 330">'
                    '<rect width="500" height="330" fill="white"/>'
                    '<g font-family="sans-serif" font-size="13" fill="#222">'
                    '<text x="45" y="25">Synthetic finite-demand example</text>'
                    '<text x="45" y="48">Expected profit (synthetic currency)</text>'
                    '<path d="M50 60 V245 H420" fill="none" stroke="#222"/>'
                    f'<polyline points="{points}" fill="none" stroke="#0072B2" stroke-width="3"/>'
                    f'{ticks}{yticks}<text x="150" y="295">Order quantity (units)</text>'
                    '</g></svg>', encoding='utf-8')


def main():
    data = json.loads(Path('data/demand.json').read_text(encoding='utf-8'))
    rows, result = solve(data)
    Path('results/results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with Path('results/profit.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['quantity', 'expected_profit'])
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, Path('figures/profit.svg'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
