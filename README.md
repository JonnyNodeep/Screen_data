# DSData MVP

OCR tables from screenshots and export parsed data to Excel.

## Setup

1. Install dependencies:
   - `poetry install`
2. Prepare environment:
   - Copy `.env.example` to `.env`
   - Fill `OPENAI_API_KEY`

## Run

`poetry run python main.py`

## Output

- Result file path is controlled by `RESULT_XLSX_PATH` (default: `result.xlsx`).
- Excel columns are exported in fixed order:
  - `ФИО`
  - `Дата рождения`
  - `Пол`
  - `Город`
  - `Телефон`
  - `Email`
