"""Domain-specific data templates."""

from ..core.types import Domain


def get_domain_templates(domain: Domain) -> dict:
    """Get the template configuration for a specific domain.

    Returns a dict with 'fields' (list of FieldSchema-compatible dicts)
    and 'description' (str).
    """
    templates = {
        Domain.TRAFFIC: {
            "description": "Intelligent transportation system data",
            "fields": [
                {"name": "road_id", "type": "string", "description": "路段标识"},
                {"name": "timestamp", "type": "datetime", "description": "时间戳"},
                {"name": "traffic_flow", "type": "float", "description": "车流量(辆/小时)"},
                {"name": "avg_speed", "type": "float", "description": "平均速度(km/h)"},
                {"name": "density", "type": "float", "description": "交通密度(辆/km)"},
                {"name": "occupancy", "type": "float", "description": "占有率(%)"},
                {"name": "weather", "type": "categorical", "description": "天气状况"},
                {"name": "is_holiday", "type": "boolean", "description": "是否节假日"},
                {"name": "hour_of_day", "type": "integer", "description": "小时(0-23)"},
                {"name": "day_of_week", "type": "integer", "description": "星期(0-6)"},
                {"name": "incident_flag", "type": "boolean", "description": "是否发生事件"},
            ],
        },
        Domain.MEDICAL: {
            "description": "Clinical/healthcare data",
            "fields": [
                {"name": "patient_id", "type": "string", "description": "患者ID"},
                {"name": "age", "type": "integer", "description": "年龄"},
                {"name": "gender", "type": "categorical", "description": "性别"},
                {"name": "blood_pressure_sys", "type": "float", "description": "收缩压(mmHg)"},
                {"name": "blood_pressure_dia", "type": "float", "description": "舒张压(mmHg)"},
                {"name": "heart_rate", "type": "float", "description": "心率(bpm)"},
                {"name": "temperature", "type": "float", "description": "体温(°C)"},
                {"name": "bmi", "type": "float", "description": "BMI"},
                {"name": "smoking", "type": "boolean", "description": "是否吸烟"},
                {"name": "diagnosis", "type": "categorical", "description": "诊断结果"},
            ],
        },
        Domain.FINANCE: {
            "description": "Financial risk control data",
            "fields": [
                {"name": "user_id", "type": "string", "description": "用户ID"},
                {"name": "age", "type": "integer", "description": "年龄"},
                {"name": "income", "type": "float", "description": "年收入"},
                {"name": "credit_score", "type": "integer", "description": "信用评分"},
                {"name": "loan_amount", "type": "float", "description": "贷款金额"},
                {"name": "loan_term_months", "type": "integer", "description": "贷款期限(月)"},
                {"name": "interest_rate", "type": "float", "description": "利率(%)"},
                {"name": "employment_length", "type": "integer", "description": "工作年限"},
                {"name": "home_ownership", "type": "categorical", "description": "房产状况"},
                {"name": "default_flag", "type": "boolean", "description": "是否违约"},
            ],
        },
        Domain.NLP: {
            "description": "NLP/Text processing data",
            "fields": [
                {"name": "text_id", "type": "string", "description": "文本ID"},
                {"name": "content", "type": "text", "description": "文本内容"},
                {"name": "label", "type": "categorical", "description": "分类标签"},
                {"name": "source", "type": "string", "description": "来源"},
                {"name": "language", "type": "categorical", "description": "语言"},
                {"name": "word_count", "type": "integer", "description": "字数"},
                {"name": "created_at", "type": "datetime", "description": "创建时间"},
            ],
        },
        Domain.IOT: {
            "description": "IoT sensor data",
            "fields": [
                {"name": "device_id", "type": "string", "description": "设备ID"},
                {"name": "timestamp", "type": "datetime", "description": "时间戳"},
                {"name": "temperature", "type": "float", "description": "温度(°C)"},
                {"name": "humidity", "type": "float", "description": "湿度(%)"},
                {"name": "pressure", "type": "float", "description": "气压(hPa)"},
                {"name": "vibration", "type": "float", "description": "振动"},
                {"name": "battery_level", "type": "float", "description": "电量(%)"},
                {"name": "signal_strength", "type": "float", "description": "信号强度(dBm)"},
                {"name": "anomaly_flag", "type": "boolean", "description": "是否异常"},
            ],
        },
        Domain.ECOMMERCE: {
            "description": "E-commerce/retail data",
            "fields": [
                {"name": "order_id", "type": "string", "description": "订单ID"},
                {"name": "user_id", "type": "string", "description": "用户ID"},
                {"name": "product_id", "type": "string", "description": "商品ID"},
                {"name": "product_category", "type": "categorical", "description": "商品类别"},
                {"name": "price", "type": "float", "description": "价格"},
                {"name": "quantity", "type": "integer", "description": "数量"},
                {"name": "total_amount", "type": "float", "description": "总金额"},
                {"name": "order_date", "type": "datetime", "description": "下单日期"},
                {"name": "payment_method", "type": "categorical", "description": "支付方式"},
                {"name": "is_returned", "type": "boolean", "description": "是否退货"},
            ],
        },
    }

    return templates.get(domain, {"description": "Custom domain", "fields": []})
