from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI


MODEL_NAME = "gpt-4.1-mini"
REQUIRED_STRING_FIELDS = ("ФИО", "Дата рождения", "Пол", "Город")
OPTIONAL_NULLABLE_FIELDS = ("Телефон", "Email")
ALL_FIELDS = REQUIRED_STRING_FIELDS + OPTIONAL_NULLABLE_FIELDS

SYSTEM_PROMPT = (
    "You extract structured records from OCR text. "
    "Return only valid JSON without markdown or comments."
)

USER_PROMPT_TEMPLATE = """Преобразуй OCR-текст в JSON-массив объектов строго с полями:
- "ФИО": string
- "Дата рождения": string
- "Пол": string
- "Город": string
- "Телефон": string | null
- "Email": string | null

Требования:
1) Ответ только JSON-массив, без пояснений и markdown.
2) Каждый элемент массива содержит все 6 полей.
3) Если телефон или email отсутствуют, ставь null.

OCR-текст:
{clean_text}
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
      ФИО, Дата рождения, Пол, Город, Телефон, Email
    - Телефон/Email may be null; other fields must be non-empty strings.
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
