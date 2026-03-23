from __future__ import annotations

import os
from pathlib import Path


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


def main() -> None:
    config = get_runtime_config()
    has_api_key = bool(config["OPENAI_API_KEY"])

    print("DSData MVP bootstrap started.")
    print(f"Images directory: {config['IMAGES_DIR']}")
    print(f"Result path: {config['RESULT_XLSX_PATH']}")
    print(f"OpenAI API key configured: {'yes' if has_api_key else 'no'}")


if __name__ == "__main__":
    main()
