import ast
import json
from pathlib import Path

import black

NOTEBOOK = Path("mean_flow_experiment1.ipynb")
LINE_LENGTH = 88


def get_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.rstrip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip() + "\n",
    }


def fmt(source):
    source = source.strip("\n") + "\n"
    try:
        return black.format_str(source, mode=black.Mode(line_length=LINE_LENGTH))
    except Exception:
        return source


def kind(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "definition"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "assignment"
    if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)):
        return "control"
    return "execution"


def ast_segments(source):
    formatted = fmt(source)
    try:
        tree = ast.parse(formatted)
    except SyntaxError:
        return [formatted]

    lines = formatted.splitlines()
    chunks = []
    current = []
    current_kind = None
    current_nonblank = 0

    def flush():
        nonlocal current, current_kind, current_nonblank
        if current:
            text = "\n".join(current).strip()
            if text:
                chunks.append(fmt(text))
        current = []
        current_kind = None
        current_nonblank = 0

    for node in tree.body:
        block = lines[node.lineno - 1 : node.end_lineno]
        node_kind = kind(node)
        block_nonblank = sum(bool(line.strip()) for line in block)

        split = False
        if current:
            if node_kind == "definition" or current_kind == "definition":
                split = True
            elif node_kind != current_kind and {node_kind, current_kind} != {
                "assignment",
                "execution",
            }:
                split = True
            elif current_nonblank + block_nonblank > 16:
                split = True

        if split:
            flush()

        if not current:
            current_kind = node_kind

        current.extend(block)
        current_nonblank += block_nonblank

        if node_kind in {"definition", "control"} and current_nonblank >= 12:
            flush()

    flush()
    return chunks or [formatted]


def split_magic(source):
    magic = []
    python_lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(("!", "%")) and not python_lines:
            magic.append(line)
        else:
            python_lines.append(line)

    parts = []
    if magic:
        parts.append("\n".join(magic).strip() + "\n")
    if any(line.strip() for line in python_lines):
        parts.extend(ast_segments("\n".join(python_lines)))
    return parts


def setup_cells(source):
    if "TRAIN_STEPS" not in source or "from datasets import load_dataset" not in source:
        return None

    lines = source.splitlines()
    installs = [line for line in lines if line.lstrip().startswith(("!", "%"))]
    imports = [line for line in lines if line.startswith("import ") or line.startswith("from ")]
    rest = [line for line in lines if line not in installs + imports and line.strip()]

    runtime_keys = (
        "SEED",
        "random.seed",
        "np.random.seed",
        "torch.manual_seed",
        "torch.backends",
        "DEVICE",
        "AMP",
        "CHECKPOINT_DIR",
        "os.makedirs",
        "print(",
        "if DEVICE",
        "    print(",
    )
    runtime = [line for line in rest if line.startswith(runtime_keys)]
    training = [line for line in rest if line not in runtime]

    result = []
    if installs:
        result += [
            md("### 0-1-1. Install dependencies\n\nColab에서 필요한 패키지만 설치한다."),
            code("\n".join(installs)),
        ]
    result += [
        md("### 0-1-2. Imports\n\n표준 라이브러리, 분석 도구, PyTorch 모듈을 불러온다."),
        code("\n".join(imports)),
        md("### 0-1-3. Reproducibility · runtime\n\nseed, device, TF32, checkpoint 경로만 설정한다."),
        code(fmt("\n".join(runtime))),
        md("### 0-1-4. Training configuration\n\n학습 budget과 MeanFlow sampler/loss hyperparameter를 한 곳에 둔다."),
        code(fmt("\n".join(training))),
    ]
    return result


def needs_split(source):
    lines = source.splitlines()
    nonblank = [line for line in lines if line.strip()]
    max_len = max((len(line) for line in lines), default=0)
    blank_ratio = (len(lines) - len(nonblank)) / max(len(lines), 1)
    try:
        tree = ast.parse(source)
        defs = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
    except SyntaxError:
        defs = 0
    return (
        len(nonblank) > 18
        or max_len > 100
        or (blank_ratio < 0.08 and len(nonblank) >= 12)
        or defs > 1
        or ";" in source
    )


def audit(cells):
    issues = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = get_source(cell)
        lines = source.splitlines()
        nonblank = [line for line in lines if line.strip()]
        max_len = max((len(line) for line in lines), default=0)
        blank_ratio = (len(lines) - len(nonblank)) / max(len(lines), 1)
        try:
            tree = ast.parse(source)
            defs = sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for node in tree.body
            )
        except SyntaxError:
            defs = 0

        reasons = []
        if max_len > 100:
            reasons.append(f"max_line={max_len}")
        if ";" in source:
            reasons.append("semicolon")
        if defs > 1:
            reasons.append(f"top_defs={defs}")
        if len(nonblank) >= 20 and blank_ratio < 0.08:
            reasons.append(f"dense={len(nonblank)}")
        if reasons:
            first = next((line.strip() for line in lines if line.strip()), "")[:80]
            issues.append((index, reasons, first))
    return issues


nb = json.loads(NOTEBOOK.read_text())
new_cells = []

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        new_cells.append(cell)
        continue

    source = get_source(cell)
    special = setup_cells(source)
    if special is not None:
        new_cells.extend(special)
        continue

    parts = split_magic(source) if needs_split(source) else [fmt(source)]
    for part in parts:
        new_cells.append(code(part))

nb["cells"] = new_cells
issues = audit(new_cells)
print("cells after rewrite:", len(new_cells))
print("remaining readability issues:", len(issues))
for item in issues:
    print(item)
if issues:
    raise SystemExit("readability audit failed")

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
