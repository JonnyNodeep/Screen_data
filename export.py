from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

EXPORT_COLUMNS = ["ФИО", "Дата рождения", "Пол", "Город", "Телефон", "Email"]


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
    frame.to_excel(target_path, index=False, engine="openpyxl")

    if logger is not None:
        logger.info(
            "Exported %s records to Excel: %s",
            len(frame.index),
            target_path,
        )

    return target_path
