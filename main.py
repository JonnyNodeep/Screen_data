from __future__ import annotations

from datetime import datetime
import os
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from export import export_records_to_excel
from gpt_parser import parse_with_gpt
from ocr import extract_raw_text_with_meta, load_image_paths
from utils import preprocess_text, run_preprocess_smoke_check


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = "images"
DEFAULT_RESULT_XLSX_PATH = "result.xlsx"


def load_local_env_file(env_path: Path) -> None:
    """Load key/value pairs from .env into process environment."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_runtime_config() -> dict[str, str]:
    load_local_env_file(PROJECT_ROOT / ".env")
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "IMAGES_DIR": os.getenv("IMAGES_DIR", DEFAULT_IMAGES_DIR),
        "RESULT_XLSX_PATH": os.getenv("RESULT_XLSX_PATH", DEFAULT_RESULT_XLSX_PATH),
    }


def configure_logging() -> Path:
    """
    Configure root logger with both console and file handlers.

    Returns file path is not required for runtime, but can be used for reporting.
    """

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{ts}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicated logs on repeated runs.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path


def main() -> None:
    log_path = configure_logging()
    logger = logging.getLogger(__name__)
    config = get_runtime_config()
    has_api_key = bool(config["OPENAI_API_KEY"])
    image_paths = load_image_paths(config["IMAGES_DIR"], logger=logger)

    logger.info("DSData MVP bootstrap started.")
    logger.info("Images directory: %s", config["IMAGES_DIR"])
    logger.info("Result path: %s", config["RESULT_XLSX_PATH"])
    logger.info("OpenAI API key configured: %s", "yes" if has_api_key else "no")
    logger.info("Valid image files detected: %d", len(image_paths))

    total_files = len(image_paths)
    ocr_success = 0
    ocr_skipped = 0
    preprocess_success = 0
    preprocess_skipped = 0
    gpt_success = 0
    gpt_skipped = 0
    processing_errors = 0
    total_records = 0

    aggregated_records: list[dict[str, Any]] = []
    file_results: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    total_ocr_lines = 0
    total_gpt_records = 0

    for sample_input, sample_output in run_preprocess_smoke_check():
        logger.info("Preprocess smoke-check: %s -> %s", ascii(sample_input), ascii(sample_output))

    for idx, image_path in enumerate(image_paths, start=1):
        image_name = image_path.name
        try:
            logger.info(
                "PROGRESS %d/%d | %s | OCR started (may take several minutes per file)",
                idx,
                total_files,
                image_name,
            )
            raw_text, ocr_line_count = extract_raw_text_with_meta(image_path, logger=logger)
            if raw_text is None:
                ocr_skipped += 1
                file_results.append(
                    {"file": image_name, "status": "skipped", "stage": "ocr", "records": 0}
                )
                logger.warning("FILE %s | status=skipped | stage=ocr | reason=empty_or_failed", image_name)
                debug_rows.append(
                    {
                        "file": image_name,
                        "ocr_line_count": ocr_line_count,
                        "ocr_raw_text": None,
                        "gpt_record_count": 0,
                        "gpt_records_json": None,
                    }
                )
                continue
            ocr_success += 1
            logger.info("FILE %s | status=ok | stage=ocr", image_name)
            logger.info(
                "PROGRESS %d/%d | %s | OCR done (%d lines) — preprocessing",
                idx,
                total_files,
                image_name,
                ocr_line_count,
            )
            total_ocr_lines += ocr_line_count

            clean_text = preprocess_text(raw_text)
            if not clean_text:
                preprocess_skipped += 1
                file_results.append(
                    {
                        "file": image_name,
                        "status": "skipped",
                        "stage": "preprocess",
                        "records": 0,
                    }
                )
                logger.warning(
                    "FILE %s | status=skipped | stage=preprocess | reason=empty_clean_text",
                    image_name,
                )
                debug_rows.append(
                    {
                        "file": image_name,
                        "ocr_line_count": ocr_line_count,
                        "ocr_raw_text": raw_text,
                        "gpt_record_count": 0,
                        "gpt_records_json": None,
                    }
                )
                continue

            preprocess_success += 1
            logger.info("FILE %s | status=ok | stage=preprocess", image_name)
            logger.info(
                "PROGRESS %d/%d | %s | calling GPT API",
                idx,
                total_files,
                image_name,
            )

            records = parse_with_gpt(
                clean_text,
                api_key=config["OPENAI_API_KEY"] or None,
                logger=logger,
            )
            if not records:
                gpt_skipped += 1
                file_results.append(
                    {"file": image_name, "status": "skipped", "stage": "gpt", "records": 0}
                )
                logger.warning("FILE %s | status=skipped | stage=gpt | reason=no_records", image_name)
                debug_rows.append(
                    {
                        "file": image_name,
                        "ocr_line_count": ocr_line_count,
                        "ocr_raw_text": raw_text,
                        "gpt_record_count": 0,
                        "gpt_records_json": None,
                    }
                )
                continue

            gpt_success += 1
            record_count = len(records)
            total_records += record_count
            aggregated_records.extend(records)
            total_gpt_records += record_count
            file_results.append(
                {"file": image_name, "status": "processed", "stage": "done", "records": record_count}
            )
            drop_rate = 1.0 - (record_count / max(1, ocr_line_count))
            logger.info(
                "METRICS file=%s ocr_lines=%s gpt_records=%s drop_rate=%.3f",
                image_name,
                ocr_line_count,
                record_count,
                drop_rate,
            )
            debug_rows.append(
                {
                    "file": image_name,
                    "ocr_line_count": ocr_line_count,
                    "ocr_raw_text": raw_text,
                    "gpt_record_count": record_count,
                    "gpt_records_json": records,
                }
            )
            logger.info(
                "FILE %s | status=processed | stage=done | records=%s",
                image_name,
                record_count,
            )
        except Exception as exc:
            processing_errors += 1
            file_results.append(
                {
                    "file": image_name,
                    "status": "error",
                    "stage": "pipeline",
                    "records": 0,
                    "reason": str(exc),
                }
            )
            logger.exception("FILE %s | status=error | stage=pipeline", image_name)

    logger.info("Per-file processing summary (%s files):", len(file_results))
    for item in file_results:
        reason = item.get("reason")
        logger.info(
            "SUMMARY file=%s status=%s stage=%s records=%s%s",
            item["file"],
            item["status"],
            item["stage"],
            item["records"],
            f" | reason={reason}" if reason else "",
        )

    logger.info("Total input files: %d", total_files)
    logger.info("OCR successful files: %d", ocr_success)
    logger.info("OCR skipped files: %d", ocr_skipped)
    logger.info("Preprocessing successful files: %d", preprocess_success)
    logger.info("Preprocessing skipped files: %d", preprocess_skipped)
    logger.info("GPT successful files: %d", gpt_success)
    logger.info("GPT skipped files: %d", gpt_skipped)
    logger.info("Pipeline errors: %d", processing_errors)
    logger.info("Total aggregated records: %d", total_records)
    logger.info("Run log file: %s", log_path)
    if total_ocr_lines:
        total_drop_rate = 1.0 - (total_gpt_records / max(1, total_ocr_lines))
        logger.info(
            "METRICS TOTAL ocr_lines=%s gpt_records=%s drop_rate=%.3f",
            total_ocr_lines,
            total_gpt_records,
            total_drop_rate,
        )

    try:
        exported_path = export_records_to_excel(
            aggregated_records,
            config["RESULT_XLSX_PATH"],
            logger=logger,
        )
        logger.info("Excel export completed: %s", exported_path)
    except Exception as exc:
        logger.error("Excel export failed: %s", exc)
        logger.error("Excel export failed. Check logs for details.")

    # Диагностический экспорт: 1 строка на входное изображение.
    try:
        max_len = int(os.getenv("DEBUG_OCR_TEXT_MAX_LEN", "5000"))
        for row in debug_rows:
            if isinstance(row.get("ocr_raw_text"), str):
                row["ocr_raw_text"] = row["ocr_raw_text"][:max_len]
        debug_path = PROJECT_ROOT / "debug_rows.xlsx"
        debug_frame = pd.DataFrame(debug_rows)
        if "gpt_records_json" in debug_frame.columns:
            debug_frame["gpt_records_json"] = debug_frame["gpt_records_json"].apply(
                lambda v: None if v is None else str(v)
            )
        debug_frame.to_excel(debug_path, index=False, engine="openpyxl")
        logger.info("Debug export completed: %s", debug_path)
    except Exception as exc:
        logger.warning("Debug export failed: %s", exc)


if __name__ == "__main__":
    main()
