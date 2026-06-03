"""RAG (Retrieval-Augmented Generation) dataset generator."""

from __future__ import annotations

import logging
from typing import Any

from ..core.config import DatasetConfig
from ..core.schema import DatasetSchema
from .base import BaseGenerator

logger = logging.getLogger(__name__)


class RAGDatasetGenerator(BaseGenerator):
    """Generate datasets for RAG (Retrieval-Augmented Generation) systems.

    Produces:
    - Document collections (Markdown format)
    - QA pairs: question + answer + context triplets
    - Multi-turn dialogue data
    - Retrieval evaluation triplets (query, positive_doc, negative_docs)

    The generated data is domain-aware and can be customized by topic.
    """

    _TOPIC_TEMPLATES: dict[str, list[str]] = {
        "zh": {
            "交通": [
                "城市交通流量预测与智能调度系统的设计与实现",
                "基于深度学习的短时交通流预测方法研究",
                "智慧高速公路多源数据融合与事件检测",
                "网联自动驾驶车辆协同控制策略分析",
            ],
            "医疗": [
                "基于电子病历的疾病风险预测模型构建",
                "医学影像智能诊断系统的开发与应用",
                "临床决策支持系统中知识图谱的构建方法",
                "远程医疗监护系统数据处理与分析",
            ],
            "金融": [
                "基于机器学习的信用风险评估模型",
                "高频交易中的异常检测与风控策略",
                "金融文本情感分析在市场预测中的应用",
                "反欺诈系统中的用户行为特征工程",
            ],
        },
        "en": {
            "traffic": [
                "Traffic Flow Prediction Using Deep Learning",
                "Smart Highway Multi-Source Data Fusion",
                "Autonomous Vehicle Cooperative Control",
            ],
            "medical": [
                "Disease Risk Prediction from EHR Data",
                "Medical Image Diagnosis Systems",
                "Clinical Decision Support with Knowledge Graphs",
            ],
            "finance": [
                "Machine Learning for Credit Risk Assessment",
                "Anomaly Detection in High-Frequency Trading",
                "Sentiment Analysis for Market Prediction",
            ],
        },
    }

    _QA_TEMPLATES: dict[str, list[str]] = {
        "zh": [
            "请解释{concept}的核心原理和应用场景。",
            "{concept}与传统方法相比有哪些优势和不足？",
            "在实际项目中，{concept}的关键实施步骤是什么？",
            "关于{concept}，最新的研究进展有哪些？",
            "如何评估{concept}的性能和效果？",
        ],
        "en": [
            "Explain the core principles and applications of {concept}.",
            "What are the advantages and limitations of {concept} compared to traditional methods?",
            "What are the key implementation steps for {concept}?",
            "What is the latest research progress on {concept}?",
            "How to evaluate the performance of {concept}?",
        ],
    }

    def __init__(
        self,
        config: DatasetConfig,
        schema: DatasetSchema,
        topic: str = "",
        num_docs: int = 100,
        num_qa: int = 500,
        qa_format: str = "triplet",
    ) -> None:
        super().__init__(config, schema)
        self.topic = topic
        self.num_docs = num_docs
        self.num_qa = num_qa
        self.qa_format = qa_format
        self.lang = config.language

    def _generate_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        # Generate documents
        docs = self._generate_documents()
        for i, doc in enumerate(docs):
            records.append({
                "type": "document",
                "doc_id": f"doc_{i:06d}",
                "title": doc["title"],
                "content": doc["content"],
                "topic": doc["topic"],
                "word_count": len(doc["content"]),
            })

        # Generate QA pairs
        qa_pairs = self._generate_qa_pairs(docs)
        for i, qa in enumerate(qa_pairs):
            record: dict[str, Any] = {
                "type": "qa",
                "qa_id": f"qa_{i:06d}",
                "question": qa["question"],
                "answer": qa["answer"],
                "source_doc_id": qa["doc_id"],
            }
            if self.qa_format == "triplet":
                record["context"] = qa["context"]
            records.append(record)

        return records

    def _generate_documents(self) -> list[dict[str, Any]]:
        """Generate a document collection."""
        lang_key = "zh" if self.lang == "zh" else "en"
        templates = self._TOPIC_TEMPLATES.get(lang_key, self._TOPIC_TEMPLATES["en"])

        # Find matching topic or use all
        matching = []
        for cat, titles in templates.items():
            if not self.topic or self.topic in cat or cat in self.topic:
                matching.extend(titles)

        if not matching:
            # Fallback: use all topics
            for titles in templates.values():
                matching.extend(titles)

        docs = []
        for i in range(self.num_docs):
            title_base = matching[i % len(matching)]
            title = f"{title_base} — 第{i + 1}部分" if self.lang == "zh" else f"{title_base} — Part {i + 1}"

            # Generate structured content
            sections = []
            for s in range(3):
                section_title = f"第{s + 1}节" if self.lang == "zh" else f"Section {s + 1}"
                body = self._generate_text_body(min(200, 50 + i * 2))
                sections.append(f"## {section_title}\n\n{body}")

            content = f"# {title}\n\n" + "\n\n".join(sections)
            docs.append({"title": title, "content": content, "topic": self.topic or "general"})

        return docs

    def _generate_qa_pairs(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate QA pairs from documents."""
        templates = self._QA_TEMPLATES.get(self.lang, self._QA_TEMPLATES["en"])

        qa_pairs = []
        for i in range(self.num_qa):
            doc = docs[i % len(docs)]

            # Extract key concepts from doc title
            title_words = doc["title"].replace(" ", "").replace("—", "").replace("-", "")
            concept = title_words[:20] if len(title_words) > 20 else title_words

            question_tmpl = templates[i % len(templates)]
            question = question_tmpl.format(concept=concept)

            answer = self._generate_text_body(min(300, 100 + i * 5))

            # Context: extract a relevant chunk from the document
            content = doc["content"]
            chunk_start = (i * 50) % max(1, len(content) - 200)
            context = content[chunk_start : chunk_start + 200]

            qa_pairs.append({
                "question": question,
                "answer": answer,
                "context": context,
                "doc_id": doc.get("doc_id", f"doc_{i % len(docs):06d}"),
            })

        return qa_pairs

    def _generate_text_body(self, target_length: int) -> str:
        """Generate a body of text at approximately target_length characters."""
        if self.lang == "zh":
            sentences = [
                "本文系统地阐述了相关领域的研究背景与现状。",
                "通过深入分析现有方法的优缺点，我们提出了一种改进方案。",
                "实验结果表明，所提出的方法在多个指标上取得了显著提升。",
                "该方法在实际部署中展现出了良好的鲁棒性和泛化能力。",
                "未来的研究方向包括进一步优化算法性能和扩展应用场景。",
                "与传统方法相比，该方案在效率和准确性方面均有明显优势。",
                "数据预处理阶段采用了标准化和归一化等多种技术手段。",
                "模型的训练过程使用了交叉验证和早停策略来防止过拟合。",
                "特征工程是构建高性能模型的关键环节之一。",
                "我们对不同参数配置下的模型性能进行了全面的对比分析。",
            ]
        else:
            sentences = [
                "This paper systematically examines the research background and current state.",
                "Through in-depth analysis of existing approaches, we propose an improved solution.",
                "Experimental results demonstrate significant improvements across multiple metrics.",
                "The method exhibits robustness and generalization capability in deployment.",
                "Future research directions include further optimization and broader applications.",
                "Compared to traditional methods, this approach shows clear advantages.",
                "Data preprocessing employed normalization and standardization techniques.",
                "Model training utilized cross-validation and early stopping to prevent overfitting.",
                "Feature engineering plays a crucial role in building high-performance models.",
                "We conducted comprehensive comparative analysis under different configurations.",
            ]

        body = ""
        while len(body) < target_length:
            idx = hash(f"{len(body)}") % len(sentences)
            body += sentences[idx]
        return body[:target_length]
