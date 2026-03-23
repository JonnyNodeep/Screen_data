from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_OCR_INSTANCE: Any | None = None
_OCR_LOCK = Lock()


def _get_ocr_engine() -> Any:
    """Return a lazily initialized PaddleOCR engine."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        with _OCR_LOCK:
            if _OCR_INSTANCE is None:
                from paddleocr import PaddleOCR

                _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR_INSTANCE


def _line_sort_key(item: tuple[float, float, str]) -> tuple[int, float]:
    """Sort mostly top-to-bottom then left-to-right with row bucketing."""
    y, x, _ = item
    return (round(y / 10), x)


def extract_raw_text(image_path: str | Path, logger: logging.Logger | None = None) -> str | None:
    """Extract OCR text from an image and return one merged line or None."""
    active_logger = logger or logging.getLogger(__name__)
    image = Path(image_path)
    if not image.exists() or not image.is_file():
        active_logger.error("Image path does not exist or is not a file: %s", image)
        return None

    try:
        ocr_engine = _get_ocr_engine()
        result = ocr_engine.ocr(str(image), cls=True)
    except Exception as exc:
        active_logger.error("OCR failed for %s: %s", image, exc)
        return None

    lines: list[tuple[float, float, str]] = []
    if result:
        for page in result:
            if not page:
                continue
            for item in page:
                if not item or len(item) < 2:
                    continue
                bbox = item[0]
                text_info = item[1]
                if not bbox or not text_info:
                    continue
                text = str(text_info[0]).strip()
                if not text:
                    continue
                avg_y = sum(point[1] for point in bbox) / len(bbox)
                min_x = min(point[0] for point in bbox)
                lines.append((avg_y, min_x, text))

    if not lines:
        active_logger.warning("Skipping image with empty OCR output: %s", image)
        return None

    ordered_text = [text for _, _, text in sorted(lines, key=_line_sort_key)]
    raw_text = " ".join(ordered_text).strip()

    if not raw_text:
        active_logger.warning("Skipping image with blank OCR text after merge: %s", image)
        return None

    return raw_text


def load_image_paths(images_dir: str | Path, logger: logging.Logger | None = None) -> list[Path]:
    """Return valid image paths from a directory."""
    active_logger = logger or logging.getLogger(__name__)
    directory = Path(images_dir)

    if not directory.is_absolute():
        directory = Path(__file__).resolve().parent / directory

    if not directory.exists():
        active_logger.error("Images directory does not exist: %s", directory)
        return []

    if not directory.is_dir():
        active_logger.error("Images path is not a directory: %s", directory)
        return []

    valid_paths: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue

        if entry.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            active_logger.info("Skipping unsupported file extension: %s", entry.name)
            continue

        try:
            with Image.open(entry) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            active_logger.warning("Skipping unreadable image %s: %s", entry, exc)
            continue

        valid_paths.append(entry)

    active_logger.info("Loaded %d valid image(s) from %s", len(valid_paths), directory)
    return valid_paths
