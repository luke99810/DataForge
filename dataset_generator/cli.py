"""Command-line interface for dataset-generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config import DatasetConfig
from .core.schema import DatasetSchema
from .core.types import Domain, OutputFormat, TaskType
from .extractors.document import DocumentExtractor
from .generators.from_schema import SchemaDrivenGenerator
from .generators.rag import RAGDatasetGenerator
from .generators.tabular import TabularGenerator
from .generators.timeseries import TimeseriesGenerator
from .templates import get_domain_templates
from .utils.io import write_output
from .utils.logging_utils import setup_logging
from .validators.quality import DataQualityValidator
from .validators.report import ValidationReport


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dataset-gen",
        description="Dataset Generator — High-quality synthetic data generation toolkit",
    )
    parser.add_argument("--version", action="version", version="dataset-generator 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    gen = subparsers.add_parser("generate", help="Generate a dataset")
    gen.add_argument("--domain", type=str, default="custom", help="Domain (traffic/medical/finance/nlp/iot/ecommerce)")
    gen.add_argument("--task", type=str, default="classification", help="Task type")
    gen.add_argument("--num-records", type=int, default=1000, help="Number of records")
    gen.add_argument("--format", type=str, default="csv", help="Output format (csv/json/jsonl/parquet/xlsx)")
    gen.add_argument("--output", type=str, default="./output", help="Output directory")
    gen.add_argument("--seed", type=int, default=42, help="Random seed")
    gen.add_argument("--noise", type=float, default=0.0, help="Noise ratio (0.0-1.0)")
    gen.add_argument("--missing", type=float, default=0.0, help="Missing value rate (0.0-1.0)")
    gen.add_argument("--project-name", type=str, default="dataset", help="Project name")
    gen.add_argument("--language", type=str, default="zh", choices=["zh", "en"], help="Language for text")

    # --- from-schema ---
    fs = subparsers.add_parser("from-schema", help="Generate from a schema file (JSON/YAML)")
    fs.add_argument("--schema", type=str, required=True, help="Path to schema file")
    fs.add_argument("--num-records", type=int, default=1000)
    fs.add_argument("--output", type=str, default="./output")
    fs.add_argument("--format", type=str, default="csv")
    fs.add_argument("--seed", type=int, default=42)

    # --- from-doc ---
    fd = subparsers.add_parser("from-doc", help="Extract schema from a document and generate data")
    fd.add_argument("--input", type=str, required=True, help="Path to input document")
    fd.add_argument("--num-records", type=int, default=1000)
    fd.add_argument("--output", type=str, default="./output")
    fd.add_argument("--format", type=str, default="csv")

    # --- rag ---
    rag = subparsers.add_parser("rag", help="Generate RAG dataset")
    rag.add_argument("--topic", type=str, default="general", help="Topic for documents")
    rag.add_argument("--num-docs", type=int, default=100)
    rag.add_argument("--num-qa", type=int, default=500)
    rag.add_argument("--output", type=str, default="./output")
    rag.add_argument("--language", type=str, default="zh", choices=["zh", "en"])

    # --- validate ---
    val = subparsers.add_parser("validate", help="Validate a dataset")
    val.add_argument("--data", type=str, required=True, help="Path to data file")
    val.add_argument("--schema", type=str, required=True, help="Path to schema file")
    val.add_argument("--output", type=str, default=None, help="Report output path")

    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle the 'generate' subcommand."""
    domain = Domain.from_label(args.domain)
    task = TaskType.from_label(args.task)
    fmt = OutputFormat(args.format)

    config = DatasetConfig(
        project_name=args.project_name,
        domain=domain,
        task_type=task,
        num_records=args.num_records,
        output_format=fmt,
        output_dir=args.output,
        seed=args.seed,
        noise_ratio=args.noise,
        missing_rate=args.missing,
        language=args.language,
    )

    # Load domain template
    template = get_domain_templates(domain)
    if not template["fields"]:
        template = get_domain_templates(Domain.TRAFFIC)  # fallback

    fields_data = template["fields"]
    from .core.schema import FieldSchema

    fields = [FieldSchema.from_dict(f) for f in fields_data]

    # Auto-select target field
    target_map = {
        TaskType.CLASSIFICATION: fields[-1].name if fields else None,
        TaskType.REGRESSION: fields[-1].name if fields else None,
        TaskType.TIME_SERIES_FORECAST: fields[2].name if len(fields) > 2 else None,
    }
    target = target_map.get(task)

    schema = DatasetSchema(
        name=config.project_name,
        description=template["description"],
        domain=domain.value,
        fields=fields,
        target_field=target,
    )

    # Select generator
    if task == TaskType.TIME_SERIES_FORECAST:
        gen = TimeseriesGenerator(config, schema)
    elif task == TaskType.RAG_RETRIEVAL:
        gen = RAGDatasetGenerator(config, schema, topic=args.domain)
    else:
        gen = TabularGenerator(config, schema)

    df = gen.generate()

    # Validate
    validator = DataQualityValidator(schema)
    validator.validate(df)
    report = ValidationReport.from_validator(config.project_name, validator)
    print(report.to_markdown())

    # Write output
    ext = fmt.value
    out_path = Path(args.output) / f"{config.project_name}.{ext}"
    write_output(df, out_path, fmt)

    print(f"\n Dataset generated: {out_path}")
    return 0


def cmd_from_schema(args: argparse.Namespace) -> int:
    """Handle the 'from-schema' subcommand."""
    config = DatasetConfig(
        project_name=Path(args.schema).stem,
        num_records=args.num_records,
        output_format=OutputFormat(args.format),
        output_dir=args.output,
        seed=args.seed,
    )

    gen = SchemaDrivenGenerator(args.schema, config)
    df = gen.generate()

    # Validate
    validator = DataQualityValidator(gen.schema)
    validator.validate(df)
    report = ValidationReport.from_validator(config.project_name, validator)
    print(report.to_markdown())

    # Write
    ext = args.format
    out_path = Path(args.output) / f"{config.project_name}.{ext}"
    write_output(df, out_path, OutputFormat(args.format))

    print(f"\n Dataset generated: {out_path}")
    return 0


def cmd_from_doc(args: argparse.Namespace) -> int:
    """Handle the 'from-doc' subcommand."""
    ext = Path(args.input).suffix.lower()

    if ext in (".txt", ".md", ".csv"):
        extractor = DocumentExtractor(args.input)
    else:
        print(f"Note: '{ext}' files may need the 'pdf' or 'docx' skill for extraction.")
        print("Reading as plain text...")
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            extractor = DocumentExtractor.from_text(f.read())

    fields = extractor.extract_fields()
    if not fields:
        print("No fields could be extracted from the document.")
        return 1

    print(f"\n Extracted {len(fields)} fields:")
    for f in fields:
        print(f"  - {f['name']} ({f.get('type', 'string')}): {f.get('description', '')}")

    config = DatasetConfig(
        project_name=Path(args.input).stem,
        num_records=args.num_records,
        output_format=OutputFormat(args.format),
        output_dir=args.output,
    )

    gen = SchemaDrivenGenerator.from_extracted_fields(fields, config)
    df = gen.generate()

    # Write
    out_path = Path(args.output) / f"{config.project_name}.{args.format}"
    write_output(df, out_path, OutputFormat(args.format))

    print(f"\n Dataset generated: {out_path}")
    return 0


def cmd_rag(args: argparse.Namespace) -> int:
    """Handle the 'rag' subcommand."""
    config = DatasetConfig(
        project_name=f"rag_{args.topic}",
        domain=Domain.CUSTOM,
        task_type=TaskType.RAG_RETRIEVAL,
        num_records=args.num_docs + args.num_qa,
        output_format=OutputFormat.JSONL,
        output_dir=args.output,
        language=args.language,
    )

    schema = DatasetSchema(
        name=config.project_name,
        description=f"RAG dataset for {args.topic}",
    )

    gen = RAGDatasetGenerator(
        config, schema,
        topic=args.topic,
        num_docs=args.num_docs,
        num_qa=args.num_qa,
        qa_format="triplet",
    )

    records = gen._generate_records()

    out_path = Path(args.output) / f"{config.project_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n RAG dataset generated: {out_path}")
    print(f"  Documents: {args.num_docs}")
    print(f"  QA pairs: {args.num_qa}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle the 'validate' subcommand."""
    from .utils.io import load_dataframe

    df = load_dataframe(args.data)

    with open(args.schema, "r", encoding="utf-8") as f:
        schema_data = json.load(f)
    schema = DatasetSchema.from_dict(schema_data)

    validator = DataQualityValidator(schema)
    validator.validate(df)
    report = ValidationReport.from_validator(schema.name, validator)

    md_report = report.to_markdown()
    print(md_report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_report, encoding="utf-8")
        print(f"\nReport saved to: {args.output}")

    return 0 if report.overall_status == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    setup_logging("INFO")
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    handlers = {
        "generate": cmd_generate,
        "from-schema": cmd_from_schema,
        "from-doc": cmd_from_doc,
        "rag": cmd_rag,
        "validate": cmd_validate,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
