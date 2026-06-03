# Literature Search Guide — 文献搜索指南

如何搜索学术文献以确定数据集的特征输入和输出。

---

## 搜索策略

### 1. 搜索关键词模板

#### 中文
```
"[领域] 预测 数据集 特征"
"[领域] 机器学习 输入特征 基准"
"[领域] prediction benchmark dataset"
"[领域] 深度学习 数据集 构建"
"[任务类型] 常用特征 数据集"
```

#### English
```
"[domain] prediction dataset features benchmark"
"[domain] machine learning input variables"
"[domain] deep learning feature engineering dataset"
"[domain] forecasting model feature selection"
```

### 2. 推荐搜索源

| 来源 | URL | 说明 |
|------|-----|------|
| arXiv | arxiv.org | 最新预印本 |
| Google Scholar | scholar.google.com | 综合搜索 |
| Papers With Code | paperswithcode.com | 带代码/数据集的论文 |
| Semantic Scholar | semanticscholar.org | AI 增强搜索 |
| DBLP | dblp.org | CS 论文数据库 |
| CNKI | cnki.net | 中文学术论文 |
| Kaggle | kaggle.com | 竞赛数据集参考 |

### 3. 搜索优先级

1. **Survey/Review 论文** → 获取该领域的全面特征总结
2. **Benchmark 论文** → 标准评估数据集的特征设计
3. **State-of-the-art 论文** → 最新研究中使用的特征
4. **数据论文（Data Paper）** → 专门描述数据集的论文

---

## 特征提取流程

### Step 1: 找到 3-5 篇代表性论文

搜索该领域被引量高或最新的论文，阅读摘要和方法章节。

### Step 2: 提取特征表

大多数论文会在方法论部分列出输入特征。重点关注：
- 特征名称和描述
- 特征类型（数值/类别/文本）
- 特征来源
- 预处理方法

### Step 3: 交叉验证

对比不同论文中使用的特征：
- ✅ 多篇论文共同使用的特征 → 高优先级采用
- ✅ 只有一篇论文使用的特征 → 作为可选特征
- ✅ 领域常识特征 → 即使论文中未明确列出也应包含

### Step 4: 构建特征 Schema

将提取的特征组织为结构化 Schema（参见 `references/dataset_formats.md`）。

---

## 常见领域论文速查

### 智能交通

**关键词**: traffic flow prediction, short-term traffic forecasting, spatiotemporal traffic data

**代表性论文特征**：
- 时空特征：路段ID、时间戳、地理位置
- 交通特征：车流量、平均速度、密度、占有率
- 外部特征：天气、节假日、事件
- 时间特征：小时、星期、月份

### 医疗诊断

**关键词**: disease prediction, clinical decision support, electronic health records machine learning

**代表性论文特征**：
- 人口统计：年龄、性别、种族
- 生命体征：血压、心率、体温、呼吸率
- 实验室检查：血常规、生化、影像
- 病史：既往病史、用药史、家族史

### 金融风控

**关键词**: credit scoring, default prediction, fraud detection features

**代表性论文特征**：
- 个人信息：年龄、收入、职业、教育
- 信用历史：信用分、逾期记录、负债情况
- 贷款信息：金额、期限、利率、用途
- 行为特征：消费模式、还款习惯

### NLP/文本

**关键词**: text classification features, named entity recognition dataset, text dataset construction

---

## 搜索结果解读

解读搜索结果时关注：

1. **特征数量**：该领域通常使用多少特征？
2. **特征类型比例**：数值/类别/文本的比例？
3. **数据规模**：论文中使用的数据量级？
4. **评估指标**：该任务常用哪些指标评估？
5. **公开数据集**：有没有可参考的公开数据集？

---

## 快速搜索命令

在使用 WebSearch 时，以下模板可以直接使用：

```
WebSearch: "[domain] prediction dataset input features survey"
WebSearch: "state of the art [domain] forecasting features"
WebSearch: "benchmark dataset for [task] [domain]"
```

搜索到结果后，使用 WebFetch 获取具体论文的特征列表和实验设置章节。
