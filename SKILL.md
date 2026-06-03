---
name: dataset-generator
description: "高质量数据集生成器。当用户需要生成领域数据集、为机器学习/深度学习/RAG项目构建训练数据、从文档中提取数据模式生成数据、根据研究文献确定输入输出特征后生成测试数据集时触发此技能。覆盖以下场景：用户提到生成数据集、构建数据集、训练数据、RAG数据集、测试数据、data generation、create dataset；用户提供论文/文档要求生成对应数据；用户描述研究领域（如智能交通、医疗诊断、金融风控）并需要预测/分类/回归任务的数据。支持程序化生成与网络采集两种模式，输出完整的GitHub项目结构。"
agent_created: true
---

# Dataset Generator — 专业高质量数据集生成框架

## 概述

本技能是一个完整的 Python 包（dataset_generator），提供高质量、领域适配的数据集生成能力，
适用于机器学习模型训练、RAG 系统构建、代码测试，以及帮助研究者明确输入输出结构后收集真实数据。

### 项目架构

```
dataset-generator/                     # 技能根目录
├── SKILL.md                           # 技能定义（当前文件）
├── pyproject.toml                     # 包元数据、依赖、CLI入口
├── dataset_generator/                 # Python 包
│   ├── __init__.py                    # 公共 API 导出
│   ├── __main__.py                    # python -m dataset_generator
│   ├── cli.py                         # 统一命令行界面
│   ├── core/                          # 核心模块
│   │   ├── types.py                   # 枚举：Domain, TaskType, FieldType, OutputFormat
│   │   ├── config.py                  # DatasetConfig 配置管理
│   │   └── schema.py                  # FieldSchema, DatasetSchema 定义与验证
│   ├── generators/                    # 生成器模块
│   │   ├── base.py                    # BaseGenerator 抽象基类
│   │   ├── tabular.py                 # 表格数据生成器
│   │   ├── timeseries.py              # 时序数据生成器
│   │   ├── rag.py                     # RAG 数据集生成器
│   │   └── from_schema.py             # Schema 驱动生成器
│   ├── extractors/                    # 模式提取模块
│   │   └── document.py                # 文档数据模式提取器
│   ├── validators/                    # 质量验证模块
│   │   ├── quality.py                 # 数据质量检查器
│   │   └── report.py                  # 验证报告生成器
│   ├── templates/                     # 领域模板
│   │   └── __init__.py                # 6大领域预定义模板
│   └── utils/                         # 工具模块
│       ├── random_utils.py            # 可复现随机数工具
│       ├── io.py                      # 文件读写工具
│       └── logging_utils.py           # 日志配置
├── scripts/                           # 便捷入口脚本
│   ├── generate_tabular.py            # → cli:generate
│   ├── generate_timeseries.py         # → cli:generate --task time_series_forecast
│   ├── generate_rag_dataset.py        # → cli:rag
│   ├── generate_from_schema.py        # → cli:from-schema
│   ├── data_validator.py              # → cli:validate
│   └── document_parser.py             # → cli:from-doc
├── tests/                             # 单元测试
│   ├── test_core.py                   # 核心模块测试
│   └── test_generators.py             # 生成器测试
├── references/                        # 参考文档
│   ├── domain_templates.md            # 领域数据模板详解
│   ├── dataset_formats.md             # 数据集格式规范
│   └── literature_search_guide.md     # 文献特征提取指南
└── assets/github/                     # GitHub 项目模板
    ├── README.md                      # 项目自述文件模板
    ├── LICENSE                        # MIT 许可证
    ├── .gitignore                     # Git 忽略规则
    ├── CONTRIBUTING.md                # 贡献指南
    └── requirements.txt               # Python 依赖
```

## 安装与环境配置

### 首次使用 — 安装依赖

技能加载后，首先检查并安装依赖：

```bash
cd <skill-directory>
pip install -e ".[all]"
```

或最小安装：

```bash
pip install -e .
```

### 验证安装

```bash
python -c "import dataset_generator; print(dataset_generator.__version__)"
# 输出: 1.0.0

dataset-gen --help
# 显示 CLI 帮助信息
```

## 触发场景

以下任一情况应加载此技能：
- 用户提到"生成数据集""构建数据集""制作数据""数据生成"
- 用户描述某个领域并表示需要训练/测试/验证数据
- 用户提供了 PDF/Word/MD 文档，希望从中提取数据模式并生成类似数据
- 用户说"我需要一些 RAG 数据""帮我造一些训练数据"
- 用户说"我要做一个 XX 预测模型，需要输入输出数据"

## 工作流程

### 阶段一：需求分析

从用户输入中提取以下关键信息：

1. **领域与主题**：使用 `Domain.from_label()` 智能推断
2. **任务类型**：使用 `TaskType.from_label()` 智能推断
3. **数据规模**：使用 `DatasetConfig.estimate_size()` 根据领域和复杂度自动确定
   - demo（测试）→ 100~500 条
   - prototype（验证）→ 1,000~5,000 条
   - production（正式）→ 10,000~100,000+ 条
4. **输入源**：用户描述 / 文档 / 文献搜索
5. **输出格式**：CSV / JSON / JSONL / Parquet / Excel

如果用户描述不够清晰，使用 `AskUserQuestion` 工具进行关键澄清。

### 阶段二：特征确定

#### A. 用户提供了文档（PDF/Word/MD）

1. 对于 PDF：加载 `pdf` 技能提取文本
2. 对于 Word：加载 `docx` 技能提取文本
3. 使用 `DocumentExtractor` 提取字段模式：
   ```python
   from dataset_generator.extractors import DocumentExtractor
   extractor = DocumentExtractor("path/to/doc.md")
   fields = extractor.extract_fields()
   ```
4. 确认提取的字段定义

#### B. 用户描述了研究领域（如智能交通预测）

1. 搜索研究文献（使用 WebSearch）：
   - 关键词：`"[领域] prediction dataset features benchmark"`
2. 加载 `references/literature_search_guide.md` 获取搜索策略
3. 提取特征列表并让用户确认
4. 使用 `templates.get_domain_templates()` 获取领域默认模板作为起点

#### C. 用户直接指定了字段

直接使用用户指定的字段定义。

### 阶段三：数据生成

#### CLI 方式（推荐）

```bash
# 从领域模板生成表格数据
dataset-gen generate --domain traffic --num-records 5000 --format csv --output ./output

# 从 Schema 文件生成
dataset-gen from-schema --schema schemas/traffic_schema.json --num-records 10000 --output ./output

# 从文档提取并生成
dataset-gen from-doc --input data/requirements.md --num-records 2000 --output ./output

# 生成 RAG 数据集
dataset-gen rag --topic "智能交通" --num-docs 100 --num-qa 500 --output ./output

# 验证已有数据集
dataset-gen validate --data output/dataset.csv --schema schemas/schema.json
```

#### Python API 方式

```python
from dataset_generator import DatasetConfig, DatasetSchema, FieldSchema, Domain, TaskType
from dataset_generator.generators import TabularGenerator, TimeseriesGenerator
from dataset_generator.validators import DataQualityValidator
from dataset_generator.templates import get_domain_templates

# 使用领域模板
template = get_domain_templates(Domain.TRAFFIC)
fields = [FieldSchema.from_dict(f) for f in template["fields"]]

config = DatasetConfig(
    project_name="traffic_prediction",
    domain=Domain.TRAFFIC,
    task_type=TaskType.TIME_SERIES_FORECAST,
    num_records=5000,
    seed=42,
)

schema = DatasetSchema(
    name="traffic_prediction",
    fields=fields,
    target_field="traffic_flow",
)

gen = TimeseriesGenerator(config, schema, freq="5min")
df = gen.generate()

# 验证
validator = DataQualityValidator(schema)
issues = validator.validate(df)
```

#### 网络数据采集模式

当需要真实/公开数据时：
1. 加载 `senior-crawler-engineer` 技能
2. 搜索合适的数据源
3. 爬取并清洗后使用 `DatasetSchema` 标准化

### 阶段四：质量验证

对每批生成数据执行：

```python
from dataset_generator.validators import DataQualityValidator, ValidationReport

validator = DataQualityValidator(schema)
issues = validator.validate(df)
report = ValidationReport.from_validator("dataset_name", validator)
print(report.to_markdown())

if not validator.is_valid:
    # 修复问题后重新生成
    ...
```

### 阶段五：GitHub 项目构建

使用 `assets/github/` 中的模板构建完整的开源项目结构：

```
dataset-{project-name}/
├── README.md
├── LICENSE
├── .gitignore
├── CONTRIBUTING.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── scripts/
│   ├── generate.py
│   └── sample.py
├── schemas/
│   └── schema.json
└── docs/
    └── data_dictionary.md
```

### 阶段六：交付

1. 将生成的数据集和项目文件写入工作区
2. 使用 `deliver_attachments` 交付关键文件（README、data_dictionary.md、示例数据）
3. 运行 `dataset-gen validate` 确认数据质量
4. 告知用户项目结构和使用方法

## 领域模板参考

| 领域 | 预设特征数 | 典型任务 |
|------|-----------|---------|
| 交通（traffic） | 11 | 流量预测、事件检测、路径规划 |
| 医疗（medical） | 10 | 疾病预测、诊断分类、风险评估 |
| 金融（finance） | 10 | 信用评分、欺诈检测、风险预测 |
| NLP | 7 | 文本分类、情感分析、实体识别 |
| IoT | 9 | 异常检测、预测性维护、能耗预测 |
| 电商（ecommerce） | 10 | 推荐系统、用户分群、需求预测 |

详细模板定义见 `dataset_generator/templates/__init__.py` 和 `references/domain_templates.md`。

## 开发与测试

```bash
# 运行测试
pytest tests/ -v

# 类型检查
mypy dataset_generator/

# 代码风格
ruff check dataset_generator/

# 安装开发依赖
pip install -e ".[dev]"
```

## 重要原则

1. **可复现性**：固定 seed，所有生成可精确复现
2. **真实性**：数据分布接近真实世界，避免明显合成痕迹
3. **文档完整**：每个数据集附带数据字典和 Schema 定义
4. **GitHub 规范**：遵循开源社区最佳实践
5. **隐私安全**：不包含任何真实个人信息
6. **类型安全**：完整的 type hints 和 mypy 验证
