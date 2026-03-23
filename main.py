from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from export import export_records_to_excel
from gpt_parser import parse_with_gpt
from ocr import extract_raw_text, load_image_paths
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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    config = get_runtime_config()
    has_api_key = bool(config["OPENAI_API_KEY"])
    image_paths = load_image_paths(config["IMAGES_DIR"], logger=logger)

    print("DSData MVP bootstrap started.")
    print(f"Images directory: {config['IMAGES_DIR']}")
    print(f"Result path: {config['RESULT_XLSX_PATH']}")
    print(f"OpenAI API key configured: {'yes' if has_api_key else 'no'}")
    print(f"Valid image files detected: {len(image_paths)}")

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

    for sample_input, sample_output in run_preprocess_smoke_check():
        logger.info("Preprocess smoke-check: %s -> %s", ascii(sample_input), ascii(sample_output))

    for image_path in image_paths:
        image_name = image_path.name
        try:
            raw_text = extract_raw_text(image_path, logger=logger)
            if raw_text is None:
                ocr_skipped += 1
                file_results.append(
                    {"file": image_name, "status": "skipped", "stage": "ocr", "records": 0}
                )
                logger.warning("FILE %s | status=skipped | stage=ocr | reason=empty_or_failed", image_name)
                continue
            ocr_success += 1
            logger.info("FILE %s | status=ok | stage=ocr", image_name)

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
                continue

            preprocess_success += 1
            logger.info("FILE %s | status=ok | stage=preprocess", image_name)

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
                continue

            gpt_success += 1
            record_count = len(records)
            total_records += record_count
            aggregated_records.extend(records)
            file_results.append(
                {"file": image_name, "status": "processed", "stage": "done", "records": record_count}
            )
            logger.info(
                "FILE %s | status=processed | stage=done | records=%s",
                image_name,
                record_count,
            )
        except Exception as exc:
            processing_errors += 1
            file_results.append(
                {"file": image_name, "status": "error", "stage": "pipeline", "records": 0}
            )
            logger.error("FILE %s | status=error | stage=pipeline | reason=%s", image_name, exc)

    logger.info("Per-file processing summary (%s files):", len(file_results))
    for item in file_results:
        logger.info(
            "SUMMARY file=%s status=%s stage=%s records=%s",
            item["file"],
            item["status"],
            item["stage"],
            item["records"],
        )

    print(f"Total input files: {total_files}")
    print(f"OCR successful files: {ocr_success}")
    print(f"OCR skipped files: {ocr_skipped}")
    print(f"Preprocessing successful files: {preprocess_success}")
    print(f"Preprocessing skipped files: {preprocess_skipped}")
    print(f"GPT successful files: {gpt_success}")
    print(f"GPT skipped files: {gpt_skipped}")
    print(f"Pipeline errors: {processing_errors}")
    print(f"Total aggregated records: {total_records}")

    try:
        exported_path = export_records_to_excel(
            aggregated_records,
            config["RESULT_XLSX_PATH"],
            logger=logger,
        )
        print(f"Excel export completed: {exported_path}")
    except Exception as exc:
        logger.error("Excel export failed: %s", exc)
        print("Excel export failed. Check logs for details.")


if __name__ == "__main__":
    main()
