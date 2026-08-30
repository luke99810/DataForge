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
                {"name": "road_id", "type": "string", "description": "Road segment ID", "constraints": {"pattern": "road_id_{i:06d}"}},
                {"name": "timestamp", "type": "datetime", "description": "Timestamp", "constraints": {"start": "2024-01-01", "end": "2024-12-31"}},
                {"name": "traffic_flow", "type": "float", "description": "Traffic flow (vehicles/hour)", "constraints": {"min": 0, "max": 5000, "precision": 1}},
                {"name": "avg_speed", "type": "float", "description": "Average speed (km/h)", "constraints": {"min": 0, "max": 120, "precision": 1}},
                {"name": "density", "type": "float", "description": "Traffic density (vehicles/km)", "constraints": {"min": 0, "max": 200, "precision": 1}},
                {"name": "occupancy", "type": "float", "description": "Occupancy rate (%)", "constraints": {"min": 0, "max": 100, "precision": 1}},
                {"name": "weather", "type": "categorical", "description": "Weather condition", "constraints": {"categories": ["sunny", "cloudy", "rainy", "foggy", "snowy"], "weights": [0.4, 0.3, 0.15, 0.1, 0.05]}},
                {"name": "is_holiday", "type": "boolean", "description": "Whether holiday"},
                {"name": "hour_of_day", "type": "integer", "description": "Hour of day (0-23)", "constraints": {"min": 0, "max": 23}},
                {"name": "day_of_week", "type": "integer", "description": "Day of week (0=Mon, 6=Sun)", "constraints": {"min": 0, "max": 6}},
                {"name": "incident_flag", "type": "boolean", "description": "Whether incident occurred"},
            ],
        },
        Domain.MEDICAL: {
            "description": "Clinical/healthcare data",
            "fields": [
                {"name": "patient_id", "type": "string", "description": "Patient ID", "constraints": {"pattern": "PAT_{i:06d}"}},
                {"name": "age", "type": "integer", "description": "Age", "constraints": {"min": 0, "max": 100}},
                {"name": "gender", "type": "categorical", "description": "Gender", "constraints": {"categories": ["male", "female"], "weights": [0.48, 0.52]}},
                {"name": "blood_pressure_sys", "type": "float", "description": "Systolic BP (mmHg)", "constraints": {"min": 80, "max": 200, "precision": 0}},
                {"name": "blood_pressure_dia", "type": "float", "description": "Diastolic BP (mmHg)", "constraints": {"min": 50, "max": 130, "precision": 0}},
                {"name": "heart_rate", "type": "float", "description": "Heart rate (bpm)", "constraints": {"min": 40, "max": 180, "precision": 0}},
                {"name": "temperature", "type": "float", "description": "Body temperature (°C)", "constraints": {"min": 35.0, "max": 42.0, "precision": 1}},
                {"name": "bmi", "type": "float", "description": "BMI", "constraints": {"min": 14, "max": 50, "precision": 1}},
                {"name": "smoking", "type": "boolean", "description": "Smoking status"},
                {"name": "diagnosis", "type": "categorical", "description": "Diagnosis result", "constraints": {"categories": ["healthy", "hypertension", "diabetes", "heart_disease", "other"]}},
            ],
        },
        Domain.FINANCE: {
            "description": "Financial risk control data",
            "fields": [
                {"name": "user_id", "type": "string", "description": "User ID", "constraints": {"pattern": "USR_{i:06d}"}},
                {"name": "age", "type": "integer", "description": "Age", "constraints": {"min": 18, "max": 80}},
                {"name": "income", "type": "float", "description": "Annual income (CNY)", "constraints": {"min": 30000, "max": 2000000, "precision": 0}},
                {"name": "credit_score", "type": "integer", "description": "Credit score", "constraints": {"min": 300, "max": 850}},
                {"name": "loan_amount", "type": "float", "description": "Loan amount (CNY)", "constraints": {"min": 1000, "max": 5000000, "precision": 0}},
                {"name": "loan_term_months", "type": "integer", "description": "Loan term (months)", "constraints": {"min": 3, "max": 360}},
                {"name": "interest_rate", "type": "float", "description": "Interest rate (%)", "constraints": {"min": 3.0, "max": 24.0, "precision": 2}},
                {"name": "employment_length", "type": "integer", "description": "Employment years", "constraints": {"min": 0, "max": 40}},
                {"name": "home_ownership", "type": "categorical", "description": "Home ownership", "constraints": {"categories": ["own", "mortgage", "rent", "other"]}},
                {"name": "default_flag", "type": "boolean", "description": "Default flag"},
            ],
        },
        Domain.NLP: {
            "description": "NLP/Text processing data",
            "fields": [
                {"name": "text_id", "type": "string", "description": "Text ID", "constraints": {"pattern": "TXT_{i:06d}"}},
                {"name": "content", "type": "text", "description": "Text content"},
                {"name": "label", "type": "categorical", "description": "Classification label", "constraints": {"categories": ["positive", "negative", "neutral"], "weights": [0.35, 0.30, 0.35]}},
                {"name": "source", "type": "string", "description": "Source"},
                {"name": "language", "type": "categorical", "description": "Language", "constraints": {"categories": ["zh", "en"], "weights": [0.6, 0.4]}},
                {"name": "word_count", "type": "integer", "description": "Word count", "constraints": {"min": 10, "max": 5000}},
                {"name": "created_at", "type": "datetime", "description": "Creation time", "constraints": {"start": "2023-01-01", "end": "2024-12-31"}},
            ],
        },
        Domain.IOT: {
            "description": "IoT sensor data",
            "fields": [
                {"name": "device_id", "type": "string", "description": "Device ID", "constraints": {"pattern": "IOT_{i:06d}"}},
                {"name": "timestamp", "type": "datetime", "description": "Timestamp", "constraints": {"start": "2024-01-01", "end": "2024-12-31"}},
                {"name": "temperature", "type": "float", "description": "Temperature (°C)", "constraints": {"min": -20, "max": 60, "precision": 1}},
                {"name": "humidity", "type": "float", "description": "Humidity (%)", "constraints": {"min": 0, "max": 100, "precision": 1}},
                {"name": "pressure", "type": "float", "description": "Pressure (hPa)", "constraints": {"min": 900, "max": 1100, "precision": 1}},
                {"name": "vibration", "type": "float", "description": "Vibration (mm/s)", "constraints": {"min": 0, "max": 50, "precision": 2}},
                {"name": "battery_level", "type": "float", "description": "Battery (%)", "constraints": {"min": 0, "max": 100, "precision": 1}},
                {"name": "signal_strength", "type": "float", "description": "Signal strength (dBm)", "constraints": {"min": -110, "max": -30, "precision": 1}},
                {"name": "anomaly_flag", "type": "boolean", "description": "Anomaly flag"},
            ],
        },
        Domain.ECOMMERCE: {
            "description": "E-commerce/retail data",
            "fields": [
                {"name": "order_id", "type": "string", "description": "Order ID", "constraints": {"pattern": "ORD_{i:08d}"}},
                {"name": "user_id", "type": "string", "description": "User ID", "constraints": {"pattern": "USR_{i:06d}"}},
                {"name": "product_id", "type": "string", "description": "Product ID", "constraints": {"pattern": "SKU_{i:06d}"}},
                {"name": "product_category", "type": "categorical", "description": "Product category", "constraints": {"categories": ["electronics", "clothing", "food", "household", "beauty", "sports", "books", "toys"]}},
                {"name": "price", "type": "float", "description": "Price (CNY)", "constraints": {"min": 1, "max": 50000, "precision": 2}},
                {"name": "quantity", "type": "integer", "description": "Quantity", "constraints": {"min": 1, "max": 100}},
                {"name": "total_amount", "type": "float", "description": "Total amount (CNY)", "constraints": {"min": 1, "max": 100000, "precision": 2}},
                {"name": "order_date", "type": "datetime", "description": "Order date", "constraints": {"start": "2024-01-01", "end": "2024-12-31"}},
                {"name": "payment_method", "type": "categorical", "description": "Payment method", "constraints": {"categories": ["wechat", "alipay", "credit_card", "debit_card", "cod"]}},
                {"name": "is_returned", "type": "boolean", "description": "Return flag"},
            ],
        },
        Domain.LOGISTICS: {
            "description": "Smart logistics and supply chain data",
            "fields": [
                {"name": "shipment_id", "type": "string", "description": "Shipment ID", "constraints": {"pattern": "SHIP_{i:06d}"}},
                {"name": "origin_warehouse", "type": "categorical", "description": "Origin warehouse", "constraints": {"categories": ["WH-BJ", "WH-SH", "WH-GZ", "WH-CD", "WH-WH", "WH-SZ", "WH-HZ", "WH-NJ"]}},
                {"name": "dest_warehouse", "type": "categorical", "description": "Destination warehouse", "constraints": {"categories": ["WH-BJ", "WH-SH", "WH-GZ", "WH-CD", "WH-WH", "WH-SZ", "WH-HZ", "WH-NJ"]}},
                {"name": "distance_km", "type": "float", "description": "Distance (km)", "constraints": {"min": 50, "max": 3500, "precision": 1}},
                {"name": "weight_kg", "type": "float", "description": "Package weight (kg)", "constraints": {"min": 0.5, "max": 5000, "precision": 2}},
                {"name": "package_type", "type": "categorical", "description": "Package type", "constraints": {"categories": ["standard", "fragile", "refrigerated", "hazardous", "oversized"]}},
                {"name": "priority_level", "type": "categorical", "description": "Priority", "constraints": {"categories": ["normal", "express", "same_day"], "weights": [0.6, 0.3, 0.1]}},
                {"name": "weather_condition", "type": "categorical", "description": "Weather", "constraints": {"categories": ["clear", "rain", "fog", "snow", "storm"], "weights": [0.45, 0.25, 0.15, 0.1, 0.05]}},
                {"name": "warehouse_load_pct", "type": "float", "description": "Warehouse load (%)", "constraints": {"min": 10, "max": 100, "precision": 1}},
                {"name": "estimated_delivery_hours", "type": "float", "description": "Estimated delivery (hours)", "constraints": {"min": 2, "max": 200, "precision": 1}},
                {"name": "is_delayed", "type": "boolean", "description": "Delayed flag"},
            ],
        },
    }

    return templates.get(domain, {"description": "Custom domain", "fields": []})
