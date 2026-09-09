import ast
import json
from pathlib import Path

NOTEBOOK = Path("mean_flow_experiment1.ipynb")


def text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_text(cell, source):
    cell["source"] = source.rstrip() + "\n"


def replace_once(source, old, new):
    if old in source:
        return source.replace(old, new, 1), 1
    return source, 0


def polish(source):
    changes = 0

    if "def meanflow_outputs" in source:
        source, c = replace_once(
            source,
            "def meanflow_outputs(model, clean_images, labels=None):\n    batch_size = clean_images.size(0)",
            "def meanflow_outputs(model, clean_images, labels=None):\n    # Sample a training path and classifier-free condition.\n    batch_size = clean_images.size(0)",
        )
        changes += c
        source, c = replace_once(
            source,
            "    average_velocity, total_derivative = jvp(",
            "    # Differentiate the model output along the MeanFlow path direction.\n    average_velocity, total_derivative = jvp(",
        )
        changes += c
        source, c = replace_once(
            source,
            "    target = (velocity - (t - r)",
            "    # Stop gradients through the regression target branch.\n    target = (velocity - (t - r)",
        )
        changes += c

    if "def train_meanflow" in source:
        source, c = replace_once(
            source,
            "    model.train()\n    for step in range(1, TRAIN_STEPS + 1):",
            "    model.train()\n\n    # Main optimization loop.\n    for step in range(1, TRAIN_STEPS + 1):",
        )
        changes += c
        source, c = replace_once(
            source,
            "        optimizer.zero_grad(set_to_none=True)\n        with torch.autocast(",
            "        optimizer.zero_grad(set_to_none=True)\n\n        # Forward pass under mixed precision when CUDA is available.\n        with torch.autocast(",
        )
        changes += c
        source, c = replace_once(
            source,
            "        scaler.scale(loss).backward()",
            "        # Backpropagate, clip gradients, update parameters, then refresh EMA.\n        scaler.scale(loss).backward()",
        )
        changes += c
        source, c = replace_once(
            source,
            "        should_log = step == 1",
            "        # Record compact training diagnostics.\n        should_log = step == 1",
        )
        changes += c
        source, c = replace_once(
            source,
            "        if step % SNAPSHOT_EVERY == 0:",
            "        # Save raw training snapshots for the later AlphaFlow analysis.\n        if step % SNAPSHOT_EVERY == 0:",
        )
        changes += c
        source, c = replace_once(
            source,
            "    final_path = os.path.join(CHECKPOINT_DIR, \"meanflow_ema.pt\")",
            "    # Save the final EMA weights and return all training artifacts.\n    final_path = os.path.join(CHECKPOINT_DIR, \"meanflow_ema.pt\")",
        )
        changes += c

    if "def materialize_loader" in source:
        source, c = replace_once(
            source,
            "def materialize_loader(loader):",
            "# Materialize a DataLoader into CPU tensors for repeated analysis.\ndef materialize_loader(loader):",
        )
        changes += c
        source, c = replace_once(
            source,
            "train_images_all, train_labels_all = materialize_loader(",
            "# Cache the complete train/test splits once.\ntrain_images_all, train_labels_all = materialize_loader(",
        )
        changes += c

    if "stanczuk_table = pd.DataFrame" in source:
        source = source.replace(
            "# Measure the low-noise score spectrum for every class\n\nfor label in range(10):",
            "# Plot the class-wise singular-value spectra.\nfor label in range(10):",
        )

    if "display(pd.DataFrame(ventura_gap_rows))" in source:
        source = source.replace(
            "# Estimate score-Jacobian spectra across noise levels\n\nfor t_value in VENTURA_TIMES:",
            "# Plot normalized score-Jacobian spectra across time.\nfor t_value in VENTURA_TIMES:",
        )

    if "frequency_table = pd.DataFrame" in source:
        source = source.replace(
            "# Measure frequency-domain behavior over the chosen conditions\n\nfor band_name in",
            "# Plot wavelet-band RMS across generation time.\nfor band_name in",
        )
        source = source.replace(
            "# Frequency-domain helper routines\n\ndef reconstruct_low_and_high",
            "# Reconstruct low- and high-frequency components for visual inspection.\ndef reconstruct_low_and_high",
        )

    if "def nearest_two_l2_distances" in source:
        source = source.replace("# Evaluate memorization-related statistics\n\n", "")

    if "generated_for_memorization = sample_meanflow_one_step" in source:
        if not source.startswith("# Compute nearest-neighbor"):
            source = (
                "# Compute nearest-neighbor distance ratios and the strict memorization mask.\n"
                + source
            )
        source = source.replace(
            "most_suspicious = torch.argsort(distance_ratio)[:8]",
            "# Inspect generated samples with the smallest d1/d2 ratios.\nmost_suspicious = torch.argsort(distance_ratio)[:8]",
        )

    if "def alphaflow_cosines_on_batch" in source:
        source, c = replace_once(
            source,
            "def alphaflow_cosines_on_batch(model, clean_images):\n    model.zero_grad(set_to_none=True)",
            "def alphaflow_cosines_on_batch(model, clean_images):\n    # Build one MeanFlow batch and its JVP derivative.\n    model.zero_grad(set_to_none=True)",
        )
        changes += c
        source, c = replace_once(
            source,
            "    derivative_stopped = total_derivative.detach()",
            "    # Decompose the AlphaFlow objective into TFM, TCc, and border-FM terms.\n    derivative_stopped = total_derivative.detach()",
        )
        changes += c
        source, c = replace_once(
            source,
            "    parameters = [",
            "    # Differentiate each component with respect to the same parameter set.\n    parameters = [",
        )
        changes += c
        source, c = replace_once(
            source,
            "    flat_tfm = flatten_parameter_gradients",
            "    # Flatten parameter gradients before computing cosine similarity.\n    flat_tfm = flatten_parameter_gradients",
        )
        changes += c

    return source, changes


notebook = json.loads(NOTEBOOK.read_text())
changed_cells = 0
syntax_checked = 0

for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue

    source = text(cell)
    if source.lstrip().startswith(("!", "%")):
        continue

    polished, changes = polish(source)
    if polished != source:
        changed_cells += 1
        set_text(cell, polished)
    else:
        set_text(cell, source)

    ast.parse(text(cell))
    syntax_checked += 1

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(f"polished_code_cells={changed_cells}")
print(f"python_cells_syntax_checked={syntax_checked}")
