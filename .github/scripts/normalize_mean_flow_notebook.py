import json
from pathlib import Path

from black import FileMode, InvalidInput, format_str


NOTEBOOK_PATH = Path("mean_flow_experiment1.ipynb")
LINE_LENGTH = 88


def is_python_cell(source: str) -> bool:
    stripped = source.lstrip()
    return not stripped.startswith(("!", "%"))


def remove_blank_only_lines(source: str) -> str:
    lines = source.splitlines()
    compact_lines = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(compact_lines) + "\n"


def normalize_code_cell(source: str) -> str:
    if not source.strip() or not is_python_cell(source):
        return source

    compact = remove_blank_only_lines(source)

    try:
        formatted = format_str(
            compact,
            mode=FileMode(line_length=LINE_LENGTH),
        )
    except InvalidInput:
        return source

    return formatted.rstrip() + "\n"


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    changed_cells = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        normalized = normalize_code_cell(source)
        if normalized != source:
            cell["source"] = normalized
            changed_cells += 1

    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Normalized {changed_cells} code cells.")


if __name__ == "__main__":
    main()
