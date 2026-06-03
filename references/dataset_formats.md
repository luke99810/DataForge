# Dataset Format Specifications — 数据集格式规范

常见数据集输出格式的标准和最佳实践。

---

## 格式对比

| 格式 | 扩展名 | 优劣势 | 适用场景 |
|------|--------|--------|---------|
| CSV | .csv | 通用性好、人可读；不支持嵌套 | 表格数据、小规模数据集 |
| JSON | .json | 支持嵌套、人可读；文件大 | API 交互、配置文件 |
| JSONL | .jsonl | 支持嵌套、可流式读取 | 大规模数据集、RAG 数据 |
| Parquet | .parquet | 列式存储、压缩率高、查询快 | 大规模数据分析、ML 训练 |
| Excel | .xlsx | 人可读、多 Sheet | 汇报、非技术人员使用 |
| SQL | .sql | 可直接导入数据库 | 数据库场景 |
| Markdown | .md | 人可读、支持表格和格式化文本 | RAG 文档库、知识库 |

---

## CSV 规范

```csv
id,feature_1,feature_2,category,label,timestamp
000001,0.8532,42,A,1,2024-01-15 08:30:00
000002,0.1294,87,B,0,2024-01-15 09:15:00
```

**要求**：
- UTF-8 编码（BOM 可选）
- 第一行为列名
- 逗号分隔
- 含逗号的字段用双引号包裹
- 建议提供 `_metadata.json` 说明字段类型

---

## JSONL 规范（推荐用于大规模数据）

```jsonl
{"id": "000001", "text": "今天天气真好", "label": "正面", "metadata": {"source": "weibo", "date": "2024-01-15"}}
{"id": "000002", "text": "堵车堵了一个小时", "label": "负面", "metadata": {"source": "weibo", "date": "2024-01-15"}}
```

**要求**：
- 每行一个完整的 JSON 对象
- 无缩进（节省空间）
- UTF-8 编码
- 所有记录必须结构一致

---

## JSON Schema 格式

用于定义数据集结构的标准 JSON Schema，供 `generate_from_schema.py` 和 `generate_tabular.py` 使用：

```json
{
  "name": "traffic_flow_dataset",
  "version": "1.0.0",
  "description": "智能交通流量预测数据集",
  "expected_count": 10000,
  "fields": [
    {
      "name": "timestamp",
      "type": "datetime",
      "required": true,
      "constraints": {
        "start": "2024-01-01",
        "end": "2024-03-31",
        "format": "%Y-%m-%d %H:%M:%S"
      },
      "distribution": {
        "frequency": "5min"
      }
    },
    {
      "name": "road_segment_id",
      "type": "categorical",
      "required": true,
      "constraints": {
        "categories": ["seg_0001", "seg_0002", "seg_0003"],
        "weights": [0.4, 0.35, 0.25]
      }
    },
    {
      "name": "traffic_flow",
      "type": "integer",
      "required": true,
      "constraints": {
        "min": 0,
        "max": 5000
      },
      "distribution": {
        "distribution_type": "normal",
        "mean": 500,
        "std": 200
      }
    },
    {
      "name": "average_speed",
      "type": "float",
      "required": true,
      "constraints": {
        "min": 0,
        "max": 120,
        "precision": 1
      }
    },
    {
      "name": "weather",
      "type": "categorical",
      "required": false,
      "constraints": {
        "categories": ["晴", "阴", "雨", "雪", "雾"],
        "weights": [0.4, 0.3, 0.15, 0.05, 0.10]
      }
    },
    {
      "name": "is_holiday",
      "type": "boolean",
      "required": true,
      "constraints": {
        "prob_true": 0.07
      }
    },
    {
      "name": "traffic_flow_t1",
      "type": "integer",
      "required": true,
      "description": "预测目标: t+1时刻流量",
      "constraints": {
        "min": 0,
        "max": 5000
      }
    }
  ],
  "dependencies": [
    {
      "source": "traffic_flow",
      "target": "traffic_flow_t1",
      "factor": 0.95,
      "offset": 10,
      "noise_std": 50
    }
  ],
  "splits": {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
  }
}
```

### Schema 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | 否 | 数据集名称 |
| version | string | 否 | 版本号 |
| description | string | 否 | 数据集描述 |
| expected_count | integer | 否 | 预期记录数 |
| fields | array | 是 | 字段定义列表 |
| fields[].name | string | 是 | 字段名 |
| fields[].type | string | 是 | 类型: integer/float/string/boolean/categorical/datetime/array/object |
| fields[].required | boolean | 否 | 是否必填 |
| fields[].constraints | object | 否 | 约束条件 |
| fields[].distribution | object | 否 | 分布参数 |
| dependencies | array | 否 | 字段间依赖关系 |
| splits | object | 否 | 数据分割比例 |

---

## 文件命名规范

```
{project_name}_{subset}_{version}.{format}

示例：
traffic_flow_train_v1.csv
traffic_flow_val_v1.csv
traffic_flow_test_v1.csv
traffic_flow_schema_v1.json
```

---

## RAG 数据格式

### 文档数据 (JSONL)

```jsonl
{"doc_id": "doc_000001", "title": "交通信号控制原理", "content": "...", "category": "交通工程", "source": "textbook", "chunks": ["chunk1...", "chunk2..."]}
```

### QA 数据 (JSONL)

```jsonl
{"id": "qa_000001", "question": "什么是智能交通系统？", "answer": "智能交通系统（ITS）是...", "context": "智能交通系统（Intelligent Transportation System）...", "topic": "交通"}
```

### 多轮对话 (JSON)

```json
{
  "id": "multi_000001",
  "turns": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    {"role": "user", "content": "我想了解智能交通"},
    {"role": "assistant", "content": "好的，智能交通系统..."}
  ]
}
```
