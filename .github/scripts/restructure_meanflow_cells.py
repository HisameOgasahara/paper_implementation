import json
from pathlib import Path

NOTEBOOK = Path("mean_flow_experiment1.ipynb")


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def make_markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.rstrip() + "\n",
    }


def make_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip() + "\n",
    }


def combine_code(cells):
    parts = [source_text(cell).strip() for cell in cells if source_text(cell).strip()]
    return "\n\n".join(parts)


def strip_subheading_markdown(text, prefix):
    lines = text.splitlines()
    if lines and lines[0].startswith(prefix):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def find_heading_index(cells, heading_prefix):
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown" and source_text(cell).lstrip().startswith(heading_prefix):
            return index
    return None


def collect_range(cells, start_prefix, end_prefix=None):
    start = find_heading_index(cells, start_prefix)
    if start is None:
        return None, None
    if end_prefix is None:
        end = len(cells)
    else:
        end = find_heading_index(cells, end_prefix)
        if end is None:
            end = len(cells)
    return start, end


def compact_zero_section(cells, start_prefix, end_prefix, title, mode):
    start, end = collect_range(cells, start_prefix, end_prefix)
    block = cells[start:end]
    main_md = source_text(block[0]).strip()
    code_cells = [cell for cell in block if cell.get("cell_type") == "code"]

    extra_md = []
    for cell in block[1:]:
        if cell.get("cell_type") != "markdown":
            continue
        text = source_text(cell).strip()
        if not text:
            continue
        body = strip_subheading_markdown(text, "###")
        if body:
            extra_md.append(body)

    explanation = main_md
    if extra_md:
        explanation += "\n\n" + "\n\n".join(extra_md)

    result = [make_markdown(explanation)]

    if mode == "setup":
        install = []
        rest = []
        for cell in code_cells:
            text = source_text(cell).strip()
            if text.startswith("!") or text.startswith("%"):
                install.append(cell)
            else:
                rest.append(cell)
        if install:
            result.append(make_code(combine_code(install)))
        if rest:
            result.append(make_code(combine_code(rest)))
        return result

    if mode == "training":
        definitions = []
        execution = []
        execution_started = False
        for cell in code_cells:
            text = source_text(cell)
            if not execution_started and (
                "meanflow_model = fresh_model()" in text
                or "meanflow_ema, training_log" in text
            ):
                execution_started = True
            (execution if execution_started else definitions).append(cell)
        if definitions:
            result.append(make_code(combine_code(definitions)))
        if execution:
            result.append(make_code(combine_code(execution)))
        return result

    if mode == "verification":
        # Keep verification calculations together; sampling is handled separately.
        if code_cells:
            result.append(make_code(combine_code(code_cells)))
        return result

    if code_cells:
        result.append(make_code(combine_code(code_cells)))
    return result


def compact_diagnostic_section(block):
    # Preserve the paper explanation as one markdown cell.
    markdown_cells = [cell for cell in block if cell.get("cell_type") == "markdown"]
    explanation_parts = []
    for cell in markdown_cells:
        text = source_text(cell).strip()
        if text:
            explanation_parts.append(text)
    explanation = "\n\n".join(explanation_parts)

    code_cells = [cell for cell in block if cell.get("cell_type") == "code"]
    if not code_cells:
        return [make_markdown(explanation)]

    # Split only where the section transitions from computation to presentation.
    split_index = None
    for i, cell in enumerate(code_cells):
        text = source_text(cell)
        if "display(" in text or "plt.figure(" in text:
            split_index = i
            break

    result = [make_markdown(explanation)]
    if split_index is None or split_index == 0:
        result.append(make_code(combine_code(code_cells)))
        return result

    calculation = code_cells[:split_index]
    presentation = code_cells[split_index:]
    result.append(make_code(combine_code(calculation)))
    result.append(make_code(combine_code(presentation)))
    return result


with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

cells = notebook["cells"]
new_cells = []
index = 0

while index < len(cells):
    cell = cells[index]
    text = source_text(cell).lstrip() if cell.get("cell_type") == "markdown" else ""

    if text.startswith("## 0-1."):
        end = find_heading_index(cells, "## 0-2.")
        new_cells.extend(compact_zero_section(cells, "## 0-1.", "## 0-2.", "setup", "setup"))
        index = end
        continue

    if text.startswith("## 0-2."):
        end = find_heading_index(cells, "## 0-3.")
        new_cells.extend(compact_zero_section(cells, "## 0-2.", "## 0-3.", "data", "plain"))
        index = end
        continue

    if text.startswith("## 0-3."):
        end = find_heading_index(cells, "## 0-4.")
        new_cells.extend(compact_zero_section(cells, "## 0-3.", "## 0-4.", "model", "plain"))
        index = end
        continue

    if text.startswith("## 0-4."):
        end = find_heading_index(cells, "## 0-5.")
        new_cells.extend(compact_zero_section(cells, "## 0-4.", "## 0-5.", "objective", "plain"))
        index = end
        continue

    if text.startswith("## 0-5."):
        # Training ends at the first later 0-6/0-7 heading.
        candidates = [
            find_heading_index(cells, "## 0-6."),
            find_heading_index(cells, "### 0-6-"),
            find_heading_index(cells, "### 0-7-"),
        ]
        candidates = [value for value in candidates if value is not None and value > index]
        end = min(candidates) if candidates else len(cells)
        block = cells[index:end]
        # Reuse local logic because section boundary is irregular in the original notebook.
        code_cells = [c for c in block if c.get("cell_type") == "code"]
        extra_md = []
        for c in block[1:]:
            if c.get("cell_type") == "markdown":
                body = strip_subheading_markdown(source_text(c).strip(), "###")
                if body:
                    extra_md.append(body)
        explanation = source_text(block[0]).strip()
        if extra_md:
            explanation += "\n\n" + "\n\n".join(extra_md)
        new_cells.append(make_markdown(explanation))
        definitions, execution = [], []
        run = False
        for c in code_cells:
            if "meanflow_model = fresh_model()" in source_text(c):
                run = True
            (execution if run else definitions).append(c)
        if definitions:
            new_cells.append(make_code(combine_code(definitions)))
        if execution:
            new_cells.append(make_code(combine_code(execution)))
        index = end
        continue

    # Merge the irregular 0-6/0-7/0-8 verification/sampling area up to diagnostic #1.
    if (
        text.startswith("## 0-6.")
        or text.startswith("### 0-6-")
        or text.startswith("### 0-7-")
        or text.startswith("### 0-8-")
    ):
        diag_start = find_heading_index(cells, "# 1.")
        block = cells[index:diag_start]
        all_code = [c for c in block if c.get("cell_type") == "code"]

        verification_code = []
        sampling_code = []
        sampling_started = False
        for c in all_code:
            code = source_text(c)
            if "def sample_meanflow_one_step" in code or "generated_preview = sample_meanflow_one_step" in code:
                sampling_started = True
            (sampling_code if sampling_started else verification_code).append(c)

        verification_md_parts = []
        sampling_md_parts = []
        sampling_md_started = False
        for c in block:
            if c.get("cell_type") != "markdown":
                continue
            md = source_text(c).strip()
            if "Conditional one-step sampling" in md or "Conditional 1-step sampler" in md or "1-step 생성 결과" in md:
                sampling_md_started = True
            body = strip_subheading_markdown(md, "###")
            if not body:
                continue
            (sampling_md_parts if sampling_md_started else verification_md_parts).append(body)

        if verification_md_parts or verification_code:
            md = "## 0-6. MeanFlow 구현 검증\n\n" + "\n\n".join(verification_md_parts)
            new_cells.append(make_markdown(md))
            if verification_code:
                new_cells.append(make_code(combine_code(verification_code)))

        if sampling_md_parts or sampling_code:
            md = "## 0-7. Conditional one-step sampling\n\n" + "\n\n".join(sampling_md_parts)
            new_cells.append(make_markdown(md))
            if sampling_code:
                new_cells.append(make_code(combine_code(sampling_code)))

        index = diag_start
        continue

    if text.startswith("# ") and len(text) > 2 and text[2].isdigit():
        next_index = index + 1
        while next_index < len(cells):
            next_cell = cells[next_index]
            next_text = source_text(next_cell).lstrip() if next_cell.get("cell_type") == "markdown" else ""
            if next_text.startswith("# ") and len(next_text) > 2 and next_text[2].isdigit():
                break
            next_index += 1
        new_cells.extend(compact_diagnostic_section(cells[index:next_index]))
        index = next_index
        continue

    new_cells.append(cell)
    index += 1

# Remove accidental adjacent duplicate markdown cells and keep notebook outputs cleared only
# for merged code cells; untouched cells preserve metadata/content.
notebook["cells"] = new_cells

with NOTEBOOK.open("w", encoding="utf-8") as handle:
    json.dump(notebook, handle, ensure_ascii=False, indent=1)
    handle.write("\n")

print(f"Restructured cells: {len(cells)} -> {len(new_cells)}")
