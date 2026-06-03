# Domain-Specific Dataset Templates

各领域数据集模板参考。包含常见输入特征、输出目标和推荐数据规模。

---

## 1. 智能交通 (Intelligent Transportation)

### 交通流量预测

**输入特征**：
| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| timestamp | datetime | 连续时间 | 采样时间戳（5min/15min/1h） |
| road_segment_id | string | seg_0001~seg_9999 | 路段标识 |
| traffic_flow | integer | 0~5000 | 车流量（辆/采样间隔） |
| average_speed | float | 0~120 | 平均车速（km/h） |
| occupancy | float | 0~100 | 车道占用率（%） |
| weather | categorical | [晴, 阴, 雨, 雪, 雾] | 天气状况 |
| temperature | float | -20~45 | 温度（℃） |
| visibility | float | 0~10000 | 能见度（m） |
| is_holiday | boolean | true/false | 是否节假日 |
| hour_of_day | integer | 0~23 | 小时 |
| day_of_week | integer | 0~6 | 星期几 |

**输出目标**：
- 未来 t+1, t+3, t+6 时刻的 traffic_flow（短期/中期/长期预测）

**推荐规模**：10,000 ~ 100,000 条（覆盖至少 3 个月）

### 交通事故预测

**额外特征**：
| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| accident_count | integer | 0~100 | 事故数量 |
| road_condition | categorical | [良好, 湿滑, 积雪, 施工] | 路况 |
| congestion_level | categorical | [畅通, 缓行, 拥堵, 严重拥堵] | 拥堵等级 |

### 交通信号优化

**额外特征**：
| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| intersection_id | string | int_0001~int_9999 | 路口标识 |
| phase_duration | integer | 10~120 | 相位时长（秒） |
| queue_length | integer | 0~500 | 排队长度（辆） |
| pedestrian_count | integer | 0~200 | 行人数量 |

---

## 2. 医疗健康 (Healthcare)

### 疾病诊断预测

**输入特征**：
| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| patient_id | string | pat_000001~ | 患者标识（匿名化） |
| age | integer | 0~120 | 年龄 |
| gender | categorical | [男, 女] | 性别 |
| bmi | float | 15~50 | 身体质量指数 |
| blood_pressure_sys | integer | 80~200 | 收缩压（mmHg） |
| blood_pressure_dia | integer | 50~130 | 舒张压（mmHg） |
| heart_rate | integer | 40~200 | 心率（bpm） |
| blood_sugar | float | 2~30 | 血糖（mmol/L） |
| cholesterol | float | 2~10 | 总胆固醇（mmol/L） |
| smoking | boolean | true/false | 是否吸烟 |
| alcohol | categorical | [从不, 偶尔, 经常] | 饮酒频率 |
| exercise_freq | categorical | [从不, 每周1-2次, 每周3-5次, 每天] | 运动频率 |
| family_history | boolean | true/false | 家族病史 |

**输出目标**：
- disease_risk：疾病风险等级 [低, 中, 高]（分类）
- diagnosis：具体诊断结果（多分类）

**推荐规模**：5,000 ~ 50,000 条

### 医疗图像诊断

| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| image_id | string | img_000001~ | 图像标识 |
| image_path | string | path | 图像路径 |
| modality | categorical | [X光, CT, MRI, 超声] | 影像类型 |
| finding | categorical | [正常, 异常] | 发现 |
| lesion_type | categorical | [结节, 肿块, 钙化, 无] | 病变类型 |
| lesion_size_mm | float | 0~200 | 病变尺寸（mm） |
| confidence | float | 0~1 | 诊断置信度 |

---

## 3. 金融风控 (Financial Risk)

### 信用评估

**输入特征**：
| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| user_id | string | user_000001~ | 用户标识 |
| age | integer | 18~80 | 年龄 |
| income_annual | float | 0~10000000 | 年收入（元） |
| employment_length | integer | 0~40 | 工作年限 |
| loan_amount | float | 1000~5000000 | 贷款金额 |
| loan_term_months | integer | 3~360 | 贷款期限（月） |
| credit_history_length | integer | 0~50 | 信用历史长度（年） |
| num_credit_lines | integer | 0~50 | 信用账户数 |
| debt_to_income_ratio | float | 0~1 | 负债收入比 |
| num_late_payments | integer | 0~100 | 逾期次数 |
| num_defaults | integer | 0~10 | 违约次数 |
| housing_status | categorical | [自有, 按揭, 租房] | 住房状态 |

**输出目标**：
- credit_score：信用评分（回归，300~850）
- default_risk：违约概率（回归，0~1）
- risk_grade：风险等级（分类，A/B/C/D/E）

**推荐规模**：10,000 ~ 200,000 条

### 欺诈检测

**额外特征**：
| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| transaction_amount | float | 0.01~10000000 | 交易金额 |
| transaction_type | categorical | [转账, 消费, 取现, 缴费] | 交易类型 |
| device_type | categorical | [iOS, Android, Web, ATM] | 设备类型 |
| ip_country | categorical | [CN, US, ...] | IP所属国家 |
| time_since_last_txn | integer | 0~86400 | 距上次交易时间（秒） |
| is_new_device | boolean | true/false | 是否新设备 |
| velocity_1h | integer | 0~100 | 1小时内交易次数 |

---

## 4. NLP/文本处理

### 文本分类数据集

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | string | 样本标识 |
| text | string | 原始文本 |
| label | categorical | 分类标签 |
| text_length | integer | 文本长度 |
| language | categorical | 语言 |

**常见任务**：情感分析、主题分类、意图识别、垃圾检测

**推荐规模**：5,000 ~ 500,000 条

### 命名实体识别 (NER)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | string | 样本标识 |
| text | string | 原始文本 |
| entities | array | 实体列表 [{text, start, end, type}] |
| entity_types | array | 实体类型 [PERSON, ORG, LOC, DATE, ...] |

### 机器翻译

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | string | 样本标识 |
| source | string | 源语言文本 |
| target | string | 目标语言文本 |
| source_lang | categorical | 源语言代码 |
| target_lang | categorical | 目标语言代码 |

### RAG 知识库数据

| 字段名 | 类型 | 说明 |
|--------|------|------|
| doc_id | string | 文档标识 |
| title | string | 文档标题 |
| content | string | 文档正文 |
| category | categorical | 文档分类 |
| source | string | 来源 |
| created_at | datetime | 创建时间 |
| chunk_id | string | 分块标识 |
| chunk_text | string | 分块文本 |
| embedding_dim | integer | 嵌入维度 |

---

## 5. 工业/物联网 (Industrial IoT)

### 预测性维护

| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| machine_id | string | m_0001~ | 机器标识 |
| timestamp | datetime | - | 时间戳 |
| temperature | float | -20~200 | 温度（℃） |
| vibration_rms | float | 0~50 | 振动均方根 |
| pressure | float | 0~1000 | 压力（kPa） |
| rpm | integer | 0~10000 | 转速 |
| power_consumption | float | 0~10000 | 功耗（W） |
| operating_hours | float | 0~100000 | 累计运行时间 |
| cycles_since_maintenance | integer | 0~10000 | 自上次维护周期数 |
| anomaly_score | float | 0~1 | 异常评分 |

**输出目标**：
- failure_risk：故障概率（回归）
- remaining_useful_life：剩余使用寿命（回归）
- maintenance_needed：是否需要维护（分类）

**推荐规模**：50,000 ~ 1,000,000 条（高频采样）

---

## 6. 电商/推荐系统

### 用户行为数据

| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| user_id | string | - | 用户标识 |
| item_id | string | - | 商品标识 |
| event_type | categorical | [浏览, 收藏, 加购, 购买, 评价] | 行为类型 |
| timestamp | datetime | - | 行为时间 |
| price | float | 0.01~100000 | 商品价格 |
| category | categorical | - | 商品类目 |
| rating | integer | 1~5 | 评分 |
| session_id | string | - | 会话标识 |

**推荐规模**：100,000 ~ 10,000,000 条

---

## 通用 Schema 示例

```json
{
  "fields": [
    {"name": "id", "type": "string", "required": true},
    {"name": "feature_1", "type": "float", "constraints": {"min": 0, "max": 1}},
    {"name": "feature_2", "type": "integer", "constraints": {"min": 0, "max": 100}},
    {"name": "category", "type": "categorical", "constraints": {"categories": ["A", "B", "C"]}},
    {"name": "label", "type": "categorical", "constraints": {"categories": ["正", "负"]}}
  ],
  "dependencies": [
    {"source": "feature_1", "target": "feature_2", "factor": 100, "offset": 0, "noise_std": 5}
  ]
}
```
