from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datetime import datetime

EXPORT_COLUMNS = ["Дата рождения", "Пол", "Город", "Телефон", "Email"]


def _build_export_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame with stable column order for export."""
    if not records:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    frame = pd.DataFrame(records)
    for column in EXPORT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame.reindex(columns=EXPORT_COLUMNS)


def export_records_to_excel(
    records: list[dict[str, Any]],
    output_path: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Export parsed records to xlsx with fixed columns and order."""
    target_path = Path(output_path)
    if target_path.parent != Path("."):
        target_path.parent.mkdir(parents=True, exist_ok=True)

    frame = _build_export_dataframe(records)
    try:
        frame.to_excel(target_path, index=False, engine="openpyxl")
    except PermissionError:
        # Если файл открыт в Excel/просмотрщике, запись в `result.xlsx` может быть
        # невозможна. Тогда сохраняем в альтернативный файл.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = target_path.with_name(f"{target_path.stem}_{ts}{target_path.suffix}")
        frame.to_excel(alt_path, index=False, engine="openpyxl")
        target_path = alt_path

    if logger is not None:
        logger.info(
            "Exported %s records to Excel: %s",
            len(frame.index),
            target_path,
        )

    return target_path
