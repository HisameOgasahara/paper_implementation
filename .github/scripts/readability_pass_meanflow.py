import ast
import json
import re
from pathlib import Path

NOTEBOOK = Path("mean_flow_experiment1.ipynb")


def text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_text(cell, source):
    cell["source"] = source.rstrip() + "\n"


def insert_comment_before_line(source, needle, comment):
    """Insert a comment before the first line containing needle, preserving indentation."""
    lines = source.splitlines()
    clean_comment = comment.strip()

    for index, line in enumerate(lines):
        if needle not in line:
            continue

        indent = line[: len(line) - len(line.lstrip())]
        rendered_comment = f"{indent}{clean_comment}"

        if rendered_comment in lines:
            return source, 0

        lines.insert(index, rendered_comment)
        return "\n".join(lines), 1

    return source, 0


def normalize_spacing(source):
    """Use blank lines to separate top-level definitions and comment-delimited blocks."""
    lines = [line.rstrip() for line in source.splitlines()]
    out = []

    for line in lines:
        stripped = line.strip()
        is_top_level_boundary = (
            line == line.lstrip()
            and (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or stripped.startswith("for ")
                or stripped.startswith("with ")
            )
        )
        is_comment = stripped.startswith("#")

        if (is_top_level_boundary or is_comment) and out and out[-1].strip():
            out.append("")
        out.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


SECTION_RULES = {
    "0-1": [
        ("import copy", "# Standard-library imports"),
        ("import matplotlib.pyplot as plt", "# Third-party imports"),
        ("SEED = 42", "# Reproducibility and runtime configuration"),
        ("TRAIN_STEPS =", "# Training hyperparameters"),
    ],
    "0-2": [
        ("try:", "# Load FashionMNIST, with an explicit fallback mirror"),
        ("def collate_fashion_mnist", "# Convert images to [-1, 1] and pack class labels"),
        ("train_loader = DataLoader", "# Build training and test data loaders"),
        ("sample_images, sample_labels", "# Quick shape and value-range sanity check"),
    ],
    "0-3": [
        ("class ScalarEmbed", "# Embed scalar time variables with Fourier features and an MLP"),
        ("class DiTBlock", "# Transformer block with adaLN-Zero conditioning"),
        ("class TinyDiT", "# Conditional TinyDiT backbone"),
        ("def fresh_model", "# Construct a fresh model on the active device"),
    ],
    "0-4": [
        ("def logit_normal_times", "# Sample scalar times from the logit-normal schedule"),
        ("def sample_meanflow_times", "# Build ordered (r, t) pairs and inject diagonal r=t cases"),
        ("def linear_path", "# Linear Gaussian interpolation path"),
        ("def maybe_drop_labels", "# Classifier-free label dropout"),
        ("def meanflow_outputs", "# MeanFlow prediction and JVP target construction"),
        ("def meanflow_loss", "# Adaptive MeanFlow training loss"),
    ],
    "0-5": [
        ("class EMA", "# Maintain an exponential-moving-average model for evaluation"),
        ("def infinite_batches", "# Cycle through the training loader indefinitely"),
        ("def train_meanflow", "# Optimize the MeanFlow model and save periodic snapshots"),
        ("meanflow_model = fresh_model", "# Launch training and keep the EMA model for analysis"),
    ],
    "0-6": [
        ("clean_images, _ = next(iter(train_loader))", "# Prepare a small batch for implementation checks"),
        ("with torch.enable_grad():", "# Verify the r=t boundary identity using the JVP target"),
        ("with torch.no_grad():", "# Evaluate diagonal velocity, denoiser, and score conversions"),
        ("def sample_meanflow_one_step", "# One-step conditional MeanFlow sampler"),
        ("generated_preview = sample_meanflow_one_step", "# Generate and visualize a conditional preview batch"),
    ],
    "1.": [
        ("FASHION_CLASS_NAMES =", "# Experiment settings and class labels"),
        ("def batched_score", "# Evaluate score vectors in memory-safe batches"),
        ("def first_anchor_for_class", "# Pick one fixed anchor image for each class"),
        ("stanczuk_rows = []", "# Accumulate class-wise spectra and intrinsic-dimension estimates"),
        ("for label in range(10):", "# Measure the low-noise score spectrum for every class"),
        ("stanczuk_table =", "# Summarize the intrinsic-dimension estimates"),
    ],
    "2.": [
        ("VENTURA_TIMES =", "# Experiment settings and orthogonal probe directions"),
        ("for t_value in VENTURA_TIMES:", "# Estimate score-Jacobian spectra across noise levels"),
        ("display(pd.DataFrame(ventura_gap_rows))", "# Summarize spectral-gap locations"),
    ],
    "3.": [
        ("NIEDOBA", "# Experiment settings"),
        ("def ", "# Helper routines for empirical-denoiser comparison"),
        ("for t_value", "# Compare the learned denoiser with the empirical optimum across time"),
        ("display(", "# Summarize and visualize the denoiser error"),
    ],
    "4.": [
        ("LI_", "# Experiment settings"),
        ("def ", "# Helper routines for representation extraction and low-dimensional analysis"),
        ("for t_value", "# Track representation geometry across time"),
        ("display(", "# Summarize and visualize representation dynamics"),
    ],
    "5.": [
        ("QIAN", "# Experiment settings"),
        ("def ", "# Frequency-domain helper routines"),
        ("for ", "# Measure frequency-domain behavior over the chosen conditions"),
        ("display(", "# Summarize and visualize frequency-domain diagnostics"),
    ],
    "6.": [
        ("BUCHANAN", "# Experiment settings"),
        ("def ", "# Helper routines for nearest-neighbor and memorization diagnostics"),
        ("for ", "# Evaluate memorization-related statistics"),
        ("display(", "# Summarize and visualize memorization diagnostics"),
    ],
    "7.": [
        ("ALPHAFLOW", "# AlphaFlow diagnostic settings"),
        ("def snapshot_step_from_path", "# Recover the training step encoded in each snapshot filename"),
        ("def alphaflow_cosines_on_batch", "# Compute the two AlphaFlow gradient-cosine diagnostics on one batch"),
        ("analysis_loader_iterator = infinite_batches(train_loader)", "# Reuse the training distribution for snapshot diagnostics"),
        ("for snapshot_path in available_snapshot_paths:", "# Evaluate every available training snapshot"),
        ("analysis_model = fresh_model()", "# Restore one snapshot into a fresh analysis model"),
        ("tfm_tc_values = []", "# Collect per-batch cosine values before aggregating statistics"),
        ("for _ in range(ALPHAFLOW_BATCHES_PER_SNAPSHOT):", "# Average diagnostics over several independent mini-batches"),
        ("for pair_name, values in [", "# Record mean and tail percentiles for each cosine pair"),
        ("del analysis_model", "# Release the snapshot model before loading the next checkpoint"),
        ("alphaflow_table =", "# Summarize snapshot-wise AlphaFlow diagnostics"),
    ],
}


def section_key(markdown):
    title = markdown.lstrip()
    for number in range(1, 7):
        if title.startswith(f"## 0-{number}"):
            return f"0-{number}"
    if title.startswith("### 0-7") or title.startswith("### 0-8"):
        return "0-6"
    for number in range(1, 8):
        if title.startswith(f"# {number}."):
            return f"{number}."
    return None


def add_generic_comment(source):
    """Give long, still-uncommented cells one useful high-level signpost."""
    nonempty = [line for line in source.splitlines() if line.strip()]
    if len(nonempty) < 18 or any(line.lstrip().startswith("#") for line in source.splitlines()):
        return source, 0

    candidates = [
        ("def ", "# Helper functions"),
        ("for ", "# Main computation loop"),
        ("display(", "# Results and visualization"),
        ("plt.figure", "# Results and visualization"),
    ]
    for needle, comment in candidates:
        source, added = insert_comment_before_line(source, needle, comment)
        if added:
            return source, 1
    return source, 0


notebook = json.loads(NOTEBOOK.read_text())
current_section = None
code_cells = 0
long_cells = 0
comments_added = 0
syntax_checked = 0

for cell in notebook["cells"]:
    if cell.get("cell_type") == "markdown":
        key = section_key(text(cell))
        if key is not None:
            current_section = key
        continue

    if cell.get("cell_type") != "code":
        continue

    code_cells += 1
    source = text(cell)
    if len([line for line in source.splitlines() if line.strip()]) >= 18:
        long_cells += 1

    if source.lstrip().startswith(("!", "%")):
        set_text(cell, source)
        continue

    for needle, comment in SECTION_RULES.get(current_section, []):
        source, added = insert_comment_before_line(source, needle, comment)
        comments_added += added

    source, added = add_generic_comment(source)
    comments_added += added
    source = normalize_spacing(source)
    set_text(cell, source)

    try:
        ast.parse(source)
        syntax_checked += 1
    except SyntaxError as exc:
        raise RuntimeError(
            f"Syntax error after readability pass in section {current_section}: {exc}"
        ) from exc

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")

print(f"code_cells_scanned={code_cells}")
print(f"long_code_cells={long_cells}")
print(f"comments_added={comments_added}")
print(f"python_cells_syntax_checked={syntax_checked}")
