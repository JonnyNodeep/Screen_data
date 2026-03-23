from __future__ import annotations

import re

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")
_REPLACEMENT_CHAR = "\uFFFD"
_ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space / BOM
}

# Repeated separator/punctuation runs often appear as OCR noise.
_NOISE_SEPARATOR_RUN_RE = re.compile(r"([|~`^_=\-.,:;*])\1{2,}")
_MIN_CLEAN_TEXT_LEN = 5


def preprocess_text(raw_text: str) -> str:
    """Normalize OCR text to a stable string for GPT parsing."""
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be str, got {type(raw_text).__name__}")

    normalized = raw_text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = normalized.replace(_REPLACEMENT_CHAR, "")
    normalized = "".join(char for char in normalized if char not in _ZERO_WIDTH_CHARS)
    normalized = _CONTROL_CHARS_RE.sub("", normalized)
    normalized = _NOISE_SEPARATOR_RUN_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if len(normalized) < _MIN_CLEAN_TEXT_LEN:
        return ""
    return normalized


def run_preprocess_smoke_check() -> list[tuple[str, str]]:
    """Run simple preprocessing checks and return input/output pairs."""
    samples = [
        "Иванов\nИван\tИванович   1990",
        "Город:\x0b Москва \uFFFD  +7(999)123-45-67",
        "email:\u200b test@example.com ||||||",
    ]
    return [(sample, preprocess_text(sample)) for sample in samples]
