from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any
import re

import os
import tempfile
import statistics
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_OCR_ENGINES: dict[str, Any] = {}
_OCR_LOCK = Lock()

# Basic Cyrillic block: используем для оценки качества распознавания.
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _get_ocr_engine(lang: str) -> Any:
    """Return a lazily initialized PaddleOCR engine for `lang`."""
    global _OCR_ENGINES
    if lang not in _OCR_ENGINES:
        with _OCR_LOCK:
            if lang not in _OCR_ENGINES:
                from paddleocr import PaddleOCR

                # Улучшаем детект/пороги без изменения формата результата.
                _OCR_ENGINES[lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    text_det_limit_side_len=1280,
                    text_det_box_thresh=0.4,
                    text_rec_score_thresh=0.5,
                )
    return _OCR_ENGINES[lang]


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _extract_candidates_from_result(
    result: Any,
    *,
    score_min: float = 0.4,
) -> list[dict[str, Any]]:
    """Extract OCR candidates as box+text+score."""
    candidates: list[dict[str, Any]] = []
    if not result:
        return candidates

    for page in result:
        if not page:
            continue

        try:
            rec_texts = page["rec_texts"]
            rec_boxes = page["rec_boxes"]
        except Exception:
            continue

        # rec_scores помогают отфильтровать самый шумный OCR.
        try:
            rec_scores = page.get("rec_scores", None)
        except Exception:
            rec_scores = None

        if not rec_texts or rec_boxes is None:
            continue

        for idx, (text, box) in enumerate(zip(rec_texts, rec_boxes)):
            text_str = str(text).strip()
            if not text_str:
                continue

            if rec_scores is not None:
                try:
                    if float(rec_scores[idx]) < score_min:
                        continue
                except Exception:
                    pass

            try:
                b = box.tolist() if hasattr(box, "tolist") else list(box)
                if len(b) < 4:
                    continue
                x1, y1, x2, y2 = b[:4]
                box_xyxy = (float(x1), float(y1), float(x2), float(y2))
                score = float(rec_scores[idx]) if rec_scores is not None else score_min
                candidates.append({"box": box_xyxy, "text": text_str, "score": score})
            except Exception:
                continue

    return candidates


def _merge_candidates(
    candidates: list[dict[str, Any]],
    *,
    iou_thresh: float = 0.35,
    center_dist_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Merge overlapping OCR candidates and keep the best-scoring text."""
    merged: list[dict[str, Any]] = []
    for cand in candidates:
        box_c = cand["box"]
        score_c = float(cand.get("score", 0.0))
        matched = False

        for m in merged:
            box_m = m["box"]
            if _box_iou(box_c, box_m) >= iou_thresh:
                matched = True
            else:
                # Доп. критерий на близость центров (особенно если боксы чуть смещены).
                w_m = max(1.0, box_m[2] - box_m[0])
                h_m = max(1.0, box_m[3] - box_m[1])
                tol = center_dist_ratio * min(w_m, h_m)
                if _center_dist(box_c, box_m) <= tol:
                    matched = True

            if matched:
                if score_c >= float(m.get("score", 0.0)):
                    m["box"] = box_c
                    m["text"] = cand["text"]
                    m["score"] = score_c
                break

        if not matched:
            merged.append({"box": box_c, "text": cand["text"], "score": score_c})

    return merged


def _line_sort_key(item: tuple[float, float, str]) -> tuple[int, float]:
    """Sort mostly top-to-bottom then left-to-right with row bucketing."""
    y, x, _ = item
    return (round(y / 10), x)


def extract_raw_text_with_meta(
    image_path: str | Path, logger: logging.Logger | None = None
) -> tuple[str | None, int]:
    """Return raw OCR text and the number of detected merged text-lines."""
    active_logger = logger or logging.getLogger(__name__)
    image = Path(image_path)
    if not image.exists() or not image.is_file():
        active_logger.error("Image path does not exist or is not a file: %s", image)
        return None, 0

    tmp_path: str | None = None
    try:
        # Предобработка + временный файл (самая стабильная опция для PaddleOCR).
        pil_img = None
        try:
            pil_img = Image.open(image).convert("RGB")
            pil_img = ImageOps.autocontrast(pil_img)

            max_side = 2000
            w, h = pil_img.size
            if max(w, h) > 0:
                scale = min(max_side / w, max_side / h)
                if scale > 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    resample = (
                        getattr(Image, "Resampling", Image).LANCZOS
                        if hasattr(Image, "Resampling")
                        else Image.LANCZOS
                    )
                    pil_img = pil_img.resize((new_w, new_h), resample=resample)
        except Exception:
            pil_img = None

        try:
            if pil_img is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                pil_img.save(tmp_path, format="PNG")
                ocr_input_path = tmp_path
            else:
                ocr_input_path = str(image)
        except Exception:
            ocr_input_path = str(image)

        # Multi-pass OCR.
        # Multi-pass OCR.
        # Чтобы не получить нестабильный нативный crash, `ru` делаем опциональным.
        # По умолчанию: `en,ru` для лучшей кириллицы в поле "Город".
        # При необходимости поведение можно переопределить через env `OCR_LANGS`.
        langs_env = os.getenv("OCR_LANGS", "en,ru")
        langs = []
        for part in langs_env.split(","):
            p = part.strip().lower()
            if p in ("en", "ru"):
                langs.append(p)
        if not langs:
            langs = ["en"]

        # Pass по языкам, потом объединяем кандидаты по боксу.
        all_candidates: list[dict[str, Any]] = []
        for lang in langs:
            ocr_engine = _get_ocr_engine(lang)
            try:
                try:
                    result = ocr_engine.ocr(ocr_input_path, cls=True)
                except TypeError:
                    result = ocr_engine.ocr(ocr_input_path)
            except Exception:
                continue

            all_candidates.extend(_extract_candidates_from_result(result, score_min=0.4))

        merged = _merge_candidates(all_candidates)

        line_items: list[dict[str, Any]] = []
        for item in merged:
            x1, y1, x2, y2 = item["box"]
            avg_y = (float(y1) + float(y2)) / 2.0
            min_x = min(float(x1), float(x2))
            text_str = str(item["text"]).strip()
            if not text_str:
                continue
            line_items.append(
                {
                    "avg_y": avg_y,
                    "min_x": min_x,
                    "text": text_str,
                    "y1": float(y1),
                    "y2": float(y2),
                }
            )

        if not line_items:
            active_logger.warning("Skipping image with empty OCR output: %s", image)
            return None, 0

        # Адаптивная группировка по строкам (high-recall):
        # объединяем фрагменты в строки по близости avg_y с учётом типичной высоты бокса.
        sorted_items = sorted(line_items, key=lambda it: (it["avg_y"], it["min_x"]))
        heights = [max(1.0, it["y2"] - it["y1"]) for it in sorted_items]
        median_h = statistics.median(heights) if heights else 30.0
        row_tol = max(8.0, median_h * 0.35)

        rows: list[list[dict[str, Any]]] = []
        current_row: list[dict[str, Any]] = []
        current_y: float | None = None
        for it in sorted_items:
            if current_y is None:
                current_row = [it]
                current_y = float(it["avg_y"])
                continue

            if abs(float(it["avg_y"]) - current_y) <= row_tol:
                current_row.append(it)
                # слегка обновляем текущий Y, чтобы дрейф по пикселям не разрывал строку
                current_y = (current_y * 0.7) + (float(it["avg_y"]) * 0.3)
            else:
                rows.append(current_row)
                current_row = [it]
                current_y = float(it["avg_y"])

        if current_row:
            rows.append(current_row)

        # Собираем текст внутри строк слева направо.
        row_texts: list[str] = []
        for row in rows:
            row_sorted = sorted(row, key=lambda it: it["min_x"])
            # Убираем совсем короткие фрагменты только если в строке есть более длинные.
            # Это снижает мусор и не обрезает крайние случаи.
            texts = [it["text"] for it in row_sorted]
            if len(texts) > 1:
                max_len = max(len(t) for t in texts)
                if max_len >= 3:
                    texts = [t for t in texts if len(t) >= 2 or len(t) == max_len]
            row_texts.append(" ".join(texts).strip())

        raw_text = " ".join([t for t in row_texts if t]).strip()

        if not raw_text:
            active_logger.warning("Skipping image with blank OCR text after merge: %s", image)
            return None, 0

        return raw_text, len(line_items)
    except Exception as exc:
        active_logger.error("OCR failed for %s: %s", image, exc)
        return None, 0
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def extract_raw_text(image_path: str | Path, logger: logging.Logger | None = None) -> str | None:
    raw_text, _ = extract_raw_text_with_meta(image_path, logger=logger)
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
