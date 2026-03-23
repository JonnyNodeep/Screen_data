from __future__ import annotations

import os
import logging
from pathlib import Path

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

    ocr_success = 0
    ocr_skipped = 0
    preprocess_success = 0
    preprocess_skipped = 0
    prepared_texts: list[tuple[Path, str]] = []

    for sample_input, sample_output in run_preprocess_smoke_check():
        logger.info("Preprocess smoke-check: %s -> %s", ascii(sample_input), ascii(sample_output))

    for image_path in image_paths:
        raw_text = extract_raw_text(image_path, logger=logger)
        if raw_text is None:
            ocr_skipped += 1
            logger.warning("OCR skipped for %s", image_path)
            continue
        ocr_success += 1
        logger.info("OCR extracted text for %s", image_path)

        clean_text = preprocess_text(raw_text)
        if not clean_text:
            preprocess_skipped += 1
            logger.warning("Preprocessing skipped for %s: empty/too short clean text", image_path)
            continue

        preprocess_success += 1
        prepared_texts.append((image_path, clean_text))
        logger.info("Preprocessing succeeded for %s", image_path)

    print(f"OCR successful files: {ocr_success}")
    print(f"OCR skipped files: {ocr_skipped}")
    print(f"Preprocessing successful files: {preprocess_success}")
    print(f"Preprocessing skipped files: {preprocess_skipped}")
    print(f"Prepared texts for next stage: {len(prepared_texts)}")


if __name__ == "__main__":
    main()
