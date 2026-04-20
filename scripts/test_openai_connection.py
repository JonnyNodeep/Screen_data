"""One-off: test OpenAI API from project .env. Do not print secrets."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import load_local_env_file  # noqa: E402

load_local_env_file(PROJECT_ROOT / ".env")
key = os.getenv("OPENAI_API_KEY", "").strip()
if not key:
    print("RESULT: no OPENAI_API_KEY in .env or environment")
    sys.exit(2)

from openai import OpenAI  # noqa: E402

try:
    client = OpenAI(api_key=key, timeout=60.0)
    client.models.list()
    print("RESULT: OK models.list() completed")
except Exception as exc:
    print("RESULT: FAIL —", type(exc).__name__)
    print(str(exc)[:2000])
    sys.exit(1)
