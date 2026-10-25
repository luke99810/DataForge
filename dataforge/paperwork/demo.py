"""Build a new, explicit synthetic example; never overwrite an existing project."""
import shutil
import sys
from pathlib import Path
from . import checks
from .project import discover, init, load, save, sha, source
from .runner import run


def build(root):
    root = Path(root).resolve()
    init(root)
    save(root / 'data/demand.json', {'synthetic': True, 'demand': [0, 1, 2, 3, 4],
                                   'probability': [0.1, 0.2, 0.3, 0.3, 0.1],
                                   'price': 5, 'cost': 2, 'max_order': 4})
    source(root, 'data/demand.json', 'urn:synthetic:finite-newsvendor', 'CC0-1.0', 'synthetic teaching fixture')
    shutil.copyfile(Path(__file__).with_name('newsvendor.py'), root / 'code/newsvendor.py')
    record = run(root, [sys.executable, 'code/newsvendor.py'],
                 ['data/demand.json', 'code/newsvendor.py'],
                 ['results/results.json', 'results/profit.csv', 'figures/profit.svg'], 42)
    if record['status'] != 'pass':
        raise ValueError('Demo execution failed: ' + record['id'])
    result = load(root / 'results/results.json')
    save(root / 'claims.json', [{'id': 'C1', 'run_id': record['id'], 'artifact': 'results/results.json',
                                'field': 'optimal_q', 'value': result['optimal_q'], 'abs_tol': 0},
                               {'id': 'C2', 'run_id': record['id'], 'artifact': 'results/results.json',
                                'field': 'expected_profit', 'value': result['expected_profit'], 'abs_tol': 1e-10}])
    save(root / 'figures.json', [{'path': 'figures/profit.svg', 'run_id': record['id'],
                                'sources': ['results/results.json', 'data/demand.json']}])
    save(root / 'bibliography.json', [])
    _write_review(root, result)
    evidence = {'reviews/reflection.md': sha(root / 'reviews/reflection.md')}
    save(root / 'state.json', {'mode': 'synthetic_teaching', 'publication_ready': False,
                              'stages': [{'name': name, 'status': 'pass', 'reviewer': 'demo mechanical self-check only',
                                          'evidence': evidence} for name in checks.STAGES]})
    names = [r['path'] for r in discover(root, ['.'])]
    save(root / 'manifest.json', checks.manifest(root, names))
    report = checks.validate(root)
    save(root / 'validation.json', report)
    return {'project': str(root), 'result': result, 'validation': report}


def _write_review(root, result):
    (root / 'draft/manuscript.md').write_text(f'''# 有限离散需求下的订货决策：教学性合成算例

## 摘要
在给定的合成需求分布和价格成本假设下，对有限订货集合逐一计算期望利润。最优订货量为 {result['optimal_q']} 单位，期望利润为 {result['expected_profit']:.1f} 合成货币单位。该算例不构成实证研究或可投稿成果。

## 问题、数据与方法
需求取值为0至4，对应概率为0.1、0.2、0.3、0.3、0.1。售价为5，单位成本为2，未售产品无残值，不允许缺货延期交付。最大订货量为4。数据为人工设定的教学输入，并非抽样观测。目标函数为 E[5 min(q,D) - 2q]，对 q 属于 {{0,1,2,3,4}} 穷举。

## 结果与讨论
完整目标值见 results/profit.csv；图1显示目标函数随订货量的变化。图1由本次运行生成，来源为 data/demand.json，单位为合成货币，证据等级为合成计算。最优解只在此有限集合和假设下成立，不支持现实市场预测、因果解释或普适创新主张。这里没有统计抽样，因此不报告P值或置信区间。

## 局限与可得性
未开展现实数据标定、需求分布敏感性或外部验证。数据、脚本与运行记录随本地复现包提供；没有对外发布仓储。公式与结果在本算例中直接定义和计算，未引入需要外部支持的文献论断，因此参考文献为空；这不是正式论文综述的替代品。
''', encoding='utf-8')
    (root / 'reviews/reflection.md').write_text('''# 教学算例反思
问题与有限枚举方法一致；合成数据来源和许可已登记。目标是演示计算证据链，不声称研究空白或学术创新。
文献与全文范本：本算例不适用，未声称按期刊全文范本写作。bibliography为空且正文无外部引用。
模型核验：有限决策集合完整枚举；独立测试用字典oracle检查最优解及利润。
数值、图件和运行文件同源；没有统计抽样和因果设计。机械阶段通过只适用于教学模式。
未完成真实研究的数据、文献语义、期刊政策、伦理、作者责任和投稿批准；publication_ready=false。
不得将该算例的通过记录复制到另一项真实研究。下一研究开始前须重新建模、核验并完成反思。
''', encoding='utf-8')
