from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI


MODEL_NAME = "gpt-4.1-mini"
REQUIRED_STRING_FIELDS = ("Дата рождения", "Пол", "Город")
OPTIONAL_NULLABLE_FIELDS = ("Телефон", "Email")
ALL_FIELDS = REQUIRED_STRING_FIELDS + OPTIONAL_NULLABLE_FIELDS
_CITY_CYRILLIC_RE = re.compile(r"^[А-Яа-яЁё\s\-]+$")

SYSTEM_PROMPT = (
    "You extract structured records from OCR text. "
    "Return only valid JSON without markdown or comments."
)

USER_PROMPT_TEMPLATE = """Преобразуй OCR-текст в JSON-массив объектов строго с полями:
- "Дата рождения": string
- "Пол": string
- "Город": string
- "Телефон": string | null
- "Email": string | null

Требования:
1) Ответ только JSON-массив, без пояснений и markdown.
2) Каждый элемент массива содержит все 5 полей.
3) Если телефон или email отсутствуют, ставь null.
4) Поле "Город" возвращай только на кириллице (допустимы пробел и дефис).
5) Если "Город" распознан латиницей/смешанно, нормализуй в кириллицу.
   Примеры: "Moskva" -> "Москва", "Sankt-Peterburg" -> "Санкт-Петербург".

OCR-текст:
{clean_text}
"""


REPAIR_SYSTEM_PROMPT = (
    "You repair structured records extracted from OCR. "
    "Return only valid JSON without markdown or comments."
)

REPAIR_USER_PROMPT_TEMPLATE = """OCR-текст:
{clean_text}

Текущие записи (JSON-массив):
{records_json}

Задача:
1) НЕ меняй длину массива и НЕ добавляй/не удаляй элементы.
2) Заполни пропущенные или сомнительные значения в полях, где это нужно:
   - Для обязательных полей (`Дата рождения`, `Пол`, `Город`):
     если значение null — замени на строку (или попытайся извлечь из OCR).
     если значение пустое/слишком короткое — замени на строку.
     если поле `Город` содержит латиницу/смешанную раскладку — исправь в кириллицу.
   - Для `Телефон` и `Email`:
     если там null — попробуй заполнить, иначе оставь как есть.
3) Если информации недостаточно — оставь значение null.
4) Верни только исправленный JSON-массив.
"""


def _build_client(api_key: str | None = None) -> OpenAI:
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=resolved_key)


def _extract_text_content(response: Any) -> str:
    # Supports OpenAI Responses API output format.
    output = getattr(response, "output", None) or []
    for message in output:
        if getattr(message, "type", None) != "message":
            continue
        content_parts = getattr(message, "content", None) or []
        for part in content_parts:
            if getattr(part, "type", None) == "output_text":
                return getattr(part, "text", "") or ""
    return ""


def _validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Each JSON item must be an object")

    # В ТЗ допускаются "пустые значения" (null или пустая ячейка).
    # Поэтому для обязательных полей принимаем пустую строку и/или null,
    # чтобы программа не падала на "почти валидном" OCR.
    normalized: dict[str, Any] = {field: record.get(field) for field in ALL_FIELDS}

    for field in REQUIRED_STRING_FIELDS:
        value = normalized.get(field)
        if value is None:
            normalized[field] = None
            continue
        if not isinstance(value, str):
            raise ValueError(f"Field '{field}' must be a string or null")
        normalized[field] = value if value.strip() else None

    for field in OPTIONAL_NULLABLE_FIELDS:
        value = normalized.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Field '{field}' must be string or null")
        normalized[field] = value if value is not None else None

    return normalized


def _needs_repair(records: list[dict[str, Any]]) -> bool:
    """Decide whether a repair GPT pass is worthwhile."""
    if not records:
        return False

    for rec in records:
        # если хотя бы 2 обязательных поля null/None — ремонтируем
        null_required = sum(1 for f in REQUIRED_STRING_FIELDS if rec.get(f) is None)
        if null_required >= 2:
            return True
        city = rec.get("Город")
        if isinstance(city, str):
            city_clean = city.strip()
            if city_clean and not _CITY_CYRILLIC_RE.fullmatch(city_clean):
                return True
    return False


def _repair_with_gpt(
    *,
    client: OpenAI,
    clean_text: str,
    records: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Run a second GPT call to fill null required fields."""
    records_json = json.dumps(records, ensure_ascii=False)
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REPAIR_USER_PROMPT_TEMPLATE.format(
                    clean_text=clean_text,
                    records_json=records_json,
                ),
            },
        ],
    )
    content = _extract_text_content(response).strip()
    return _parse_and_validate_json(content)


def _parse_and_validate_json(payload: str) -> list[dict[str, Any]]:
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError("Top-level JSON must be an array")
    return [_validate_record(item) for item in parsed]


def parse_with_gpt(
    clean_text: str,
    *,
    api_key: str | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """
    Parse clean OCR text into structured records using gpt-4.1-mini.

    Contract:
    - Input: non-empty clean_text (str)
    - Output: list[dict] with fixed keys:
      Дата рождения, Пол, Город, Телефон, Email
    - Телефон/Email may be null; other fields may be null on uncertain OCR.
    """
    active_logger = logger or logging.getLogger(__name__)
    if not isinstance(clean_text, str):
        active_logger.error("GPT parsing skipped: clean_text must be str")
        return []

    trimmed = clean_text.strip()
    if not trimmed:
        active_logger.warning("GPT parsing skipped: clean_text is empty")
        return []

    try:
        client = _build_client(api_key=api_key)
    except Exception as exc:
        active_logger.error("GPT parsing configuration error: %s", exc)
        return []

    for attempt in (1, 2):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(clean_text=trimmed),
                    },
                ],
            )
            content = _extract_text_content(response).strip()
            records = _parse_and_validate_json(content)

            # selective repair: если много null в обязательных полях
            if attempt == 1 and _needs_repair(records):
                try:
                    repaired = _repair_with_gpt(
                        client=client,
                        clean_text=trimmed,
                        records=records,
                        logger=active_logger,
                    )
                    active_logger.info("GPT repair succeeded: %s records", len(repaired))
                    return repaired
                except Exception as exc:
                    active_logger.warning("GPT repair failed, keep original: %s", exc)

            active_logger.info(
                "GPT parsing succeeded (attempt %s): %s records",
                attempt,
                len(records),
            )
            return records
        except (json.JSONDecodeError, ValueError) as exc:
            active_logger.warning(
                "GPT parsing invalid JSON/schema (attempt %s/2): %s",
                attempt,
                exc,
            )
            if attempt == 2:
                active_logger.error("GPT parsing failed after retry, input skipped")
                return []
        except Exception as exc:
            active_logger.error("GPT API request failed (attempt %s/2): %s", attempt, exc)
            return []

    return []
