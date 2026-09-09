"""flow_basic의 모델 계약·학습·조건부 생성·조기 종료를 CPU에서 검증한다.

실행: .venv\Scripts\python.exe -m unittest discover -s tests -v
데이터 다운로드와 전체 학습 없이 노트북 셀의 실제 함수를 실행한다.
"""

import ast
import copy
import json
import math
import random
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "flow_basic.ipynb").read_text(encoding="utf-8"))
SOURCES = {
    tag: "".join(cell["source"])
    for cell in NOTEBOOK["cells"]
    for tag in cell.get("metadata", {}).get("tags", [])
}


def execute_definitions(source, namespace):
    source = "\n".join(
        line for line in source.splitlines() if not line.startswith(("%", "!"))
    )
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "<notebook definitions>", "exec"
        ),
        namespace,
    )


def make_namespace(folder):
    namespace = dict(
        torch=torch,
        nn=nn,
        F=F,
        jvp=jvp,
        np=np,
        pd=pd,
        plt=plt,
        copy=copy,
        math=math,
        random=random,
        time=time,
        Path=Path,
        contextmanager=contextmanager,
        display=lambda value: None,
    )
    setup = ast.parse(
        "\n".join(
            line for line in SOURCES["setup"].splitlines() if not line.startswith("%")
        )
    )
    excluded = {"DATA_ROOT", "OUTPUT_ROOT", "RUN_DIR"}
    for node in setup.body:
        if isinstance(node, ast.Assign):
            names = {
                item.id
                for target in node.targets
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
            }
            if not names.intersection(excluded):
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), "<config>", "exec"
                    ),
                    namespace,
                )
    execute_definitions(SOURCES["setup"], namespace)
    namespace.update(
        DEVICE=torch.device("cpu"),
        MODEL_DIM=32,
        MODEL_DEPTH=1,
        MODEL_HEADS=4,
        FOURIER_DIM=16,
        BATCH_SIZE=4,
        DIAG_BATCH=4,
        SAMPLE_BATCH=4,
        SAMPLES_PER_CLASS=1,
        RF_TEACHER_NFE=2,
        CTM_TEACHER_NFE=2,
        RUN_DIR=Path(folder),
        MUON_BACKEND="fallback",
    )
    for tag in (
        "architecture",
        "optimizer",
        "utilities",
        "checkpoints",
        "training_loop",
        "preflight",
        "fm_definition",
        "rf_train",
        "cm_definition",
        "ctm_definition",
        "shortcut_definition",
        "meanflow_definition",
        "comparison_geometry",
        "flow_maps",
    ):
        execute_definitions(SOURCES[tag], namespace)
    namespace.update(
        CM_SIGMA_MIN=0.002, CM_SIGMA_MAX=80.0, CM_SIGMA_DATA=0.5, CM_RHO=7.0
    )
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(2)
    namespace["loader_generator"] = torch.Generator().manual_seed(42)
    images = torch.randn(8, 1, 32, 32).clamp(-1, 1)
    labels = torch.arange(8) % namespace["NUM_CLASSES"]
    namespace["train_loader"] = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=True,
        generator=namespace["loader_generator"],
    )
    return namespace


class FlowNotebookTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="flow-basic-test-")
        self.addCleanup(self.folder.cleanup)
        self.ns = make_namespace(self.folder.name)

    def test_notebook_schema_and_all_python_cells_compile(self):
        nbformat.validate(nbformat.from_dict(NOTEBOOK))
        for index, cell in enumerate(NOTEBOOK["cells"]):
            if cell["cell_type"] == "code":
                source = "\n".join(
                    line for line in cell["source"] if not line.startswith(("%", "!"))
                )
                compile(source, f"cell_{index}", "exec")
                self.assertEqual(cell["outputs"], [])

    def test_common_preflight(self):
        self.ns["verify_common_model"]()

    def test_minimal_architecture_and_objective_parity(self):
        minimal = json.loads(
            (ROOT / "meanflow_minimal/meanflow_dit_fashion_mnist.ipynb").read_text(
                encoding="utf-8"
            )
        )
        reference = dict(self.ns)
        execute_definitions("".join(minimal["cells"][8]["source"]), reference)
        torch.manual_seed(42)
        old = reference["CondDiT"](dim=32, depth=1)
        # Match the reference's fixed eight heads and 128 Fourier dimensions.
        migrated = dict(self.ns, MODEL_HEADS=8, FOURIER_DIM=128)
        execute_definitions(SOURCES["architecture"], migrated)
        torch.manual_seed(42)
        new = migrated["CondDiT"](dim=32, depth=1)
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        with torch.no_grad():
            old.out.weight.normal_(std=0.01)
            old.blocks[0].mod[-1].weight.normal_(std=0.01)
        new.load_state_dict(old.state_dict())
        images, labels = next(iter(self.ns["train_loader"]))
        reference["model"] = old
        execute_definitions("".join(minimal["cells"][13]["source"]), reference)
        torch.manual_seed(123)
        old_tuple = reference["sample_tuple"](images)
        torch.manual_seed(123)
        new_tuple = self.ns["sample_meanflow_tuple"](images)
        for a, b in zip(old_tuple, new_tuple):
            self.assertTrue(torch.equal(a, b))
        z, velocity, r, t = old_tuple
        old_prediction, old_target = reference["mf_outputs"](z, velocity, r, t, labels)
        new_prediction, new_target = self.ns["meanflow_outputs"](
            new, z, velocity, r, t, labels
        )
        torch.testing.assert_close(old_prediction, new_prediction)
        torch.testing.assert_close(old_target, new_target)
        old_loss, _ = reference["loss_fn"](old_prediction, old_target)
        torch.manual_seed(123)
        new_loss, _ = self.ns["meanflow_loss"](new, images, labels, 1)
        torch.testing.assert_close(old_loss, new_loss)
        old_loss.backward()
        new_loss.backward()
        for old_p, new_p in zip(old.parameters(), new.parameters()):
            if old_p.grad is not None:
                torch.testing.assert_close(old_p.grad, new_p.grad)

    def test_meanflow_chain_rule_by_finite_difference_and_boundary(self):
        model = self.ns["fresh_model"]()
        with torch.no_grad():
            model.out.weight.normal_(std=0.02)
            model.fmod[-1].weight.normal_(std=0.02)
            model.blocks[0].mod[-1].weight.normal_(std=0.02)
        z = torch.randn(2, 1, 32, 32)
        velocity = torch.randn_like(z)
        labels = torch.tensor([1, 7])
        t, r = torch.full((2,), 0.7), torch.full((2,), 0.2)
        fn = lambda zz, tt, rr: model(zz, tt, tt - rr, labels)
        _, derivative = jvp(
            fn, (z, t, r), (velocity, torch.ones_like(t), torch.zeros_like(r))
        )
        eps = 1e-3
        finite = (
            fn(z + eps * velocity, t + eps, r) - fn(z - eps * velocity, t - eps, r)
        ) / (2 * eps)
        torch.testing.assert_close(derivative, finite, atol=2e-4, rtol=2e-2)
        _, target = self.ns["meanflow_outputs"](model, z, velocity, t, t, labels)
        torch.testing.assert_close(target, velocity)
        self.assertFalse(target.requires_grad)

    def test_all_objectives_backward_and_optimizer_coverage(self):
        ns = self.ns
        ns["fm"] = ns["freeze_copy"](ns["fresh_model"]())
        ns["rf1"] = ns["freeze_copy"](ns["fresh_model"]())
        # A zero-initialized teacher is the identity, so its reflow target is zero.
        # Give the teacher a nonzero field to exercise the training branch.
        with torch.no_grad():
            ns["rf1"].out.weight.normal_(std=0.01)
        ns["cm_target"] = ns["freeze_copy"](ns["fresh_model"]())
        ns["ctm_target"] = ns["freeze_copy"](ns["fresh_model"]())
        images, labels = next(iter(ns["train_loader"]))
        for name in (
            "fm_loss",
            "reflow_loss",
            "cm_loss",
            "ctm_loss",
            "shortcut_loss",
            "meanflow_loss",
        ):
            with self.subTest(objective=name):
                model = ns["fresh_model"]()
                optimizers, info = ns["build_optimizers"](model)
                self.assertIn("blocks.0.mlp.0.weight", info["muon_names"])
                self.assertNotIn("ye.weight", info["muon_names"])
                # The exact minimal split includes scalar embedding MLP matrices.
                self.assertIn("te.mlp.0.weight", info["muon_names"])
                loss, _ = ns[name](model, images, labels, 1)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(model.out.weight.grad.norm().item(), 0)
                before = model.out.weight.detach().clone()
                for optimizer in optimizers:
                    optimizer.step()
                self.assertFalse(torch.equal(before, model.out.weight))
        for target in ("fm", "rf1", "cm_target", "ctm_target"):
            self.assertTrue(all(p.grad is None for p in ns[target].parameters()))

    def test_samplers_nfe_label_propagation_and_signs(self):
        class LabelField(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, z, t, interval, labels):
                self.calls.append(labels.clone())
                return labels[:, None, None, None].float().expand_as(z) / 10

        labels = torch.tensor([1, 4, 7])
        z = torch.randn(3, 1, 32, 32)
        for sampler in (
            "sample_fm",
            "sample_cm",
            "sample_ctm",
            "sample_shortcut",
            "sample_meanflow",
        ):
            for nfe in (1, 2, 4):
                model = LabelField()
                output = self.ns[sampler](model, z.clone(), labels, nfe)
                self.assertEqual(output.shape, z.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(len(model.calls), nfe)
                self.assertTrue(all(torch.equal(call, labels) for call in model.calls))
                if sampler != "sample_cm":
                    sign = 1 if sampler == "sample_shortcut" else -1
                    torch.testing.assert_close(
                        output, z + sign * labels[:, None, None, None] / 10
                    )
        model = LabelField()
        sigma = torch.full((len(z),), self.ns["CM_SIGMA_MIN"])
        torch.testing.assert_close(self.ns["cm_f"](model, z, sigma, labels), z)

    def test_ctm_projection_preserves_input_gradient(self):
        target = self.ns["freeze_copy"](self.ns["fresh_model"]())
        z = torch.randn(2, 1, 32, 32, requires_grad=True)
        t = torch.full((2,), 0.5)
        output = self.ns["ctm_map"](
            target, z, t, torch.zeros_like(t), torch.tensor([1, 2])
        )
        output.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_generation_is_deterministic_and_rng_isolated(self):
        ns = self.ns
        model = ns["fresh_model"]().train()
        labels = [0, 1, 4, 7, 9]
        ns["seed_all"](234)
        torch_state = torch.get_rng_state().clone()
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        first = ns["generate_conditioned"](model, ns["sample_cm"], labels, 2, seed=42)
        second = ns["generate_conditioned"](model, ns["sample_cm"], labels, 2, seed=42)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(random_state, random.getstate())
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertTrue(model.training)

    def test_early_stop_keeps_20000_budget_and_saves_final_state(self):
        ns = self.ns
        ns["EARLY_STOP_AT"]["MeanFlow"] = 2
        ns["DIAG_EVERY"] = 10
        model = ns["fresh_model"]()
        with patch("builtins.print"):
            ns["train_method"](
                "MeanFlow", model, ns["meanflow_loss"], ns["sample_meanflow"]
            )
        self.assertEqual(ns["TRAIN_STEPS"], 20_000)
        self.assertEqual(ns["RUN_INFO"]["MeanFlow"]["train_steps"], 2)
        self.assertEqual(ns["RUN_INFO"]["MeanFlow"]["stop_reason"], "early_stop_at")
        path = Path(self.folder.name) / "MeanFlow" / "last.pt"
        saved = torch.load(path, weights_only=True)
        self.assertEqual(saved["step"], 2)
        self.assertEqual(saved["config"]["TRAIN_STEPS"], 20_000)
        self.assertEqual(len(saved["optimizers"]), 2)
        self.assertEqual(
            float(saved["optimizers"][0]["param_groups"][0]["lr"]), ns["MUON_LR"]
        )
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(saved["model"][name], parameter)
        self.assertEqual(
            list(pd.read_csv(path.with_name("training.csv"))["step"]), [1, 2]
        )
        with patch.object(plt, "show"):
            ns["show_conditional"](model, ns["sample_meanflow"], "MeanFlow", 1, step=2)
        self.assertTrue((path.parent / "conditional_00002_nfe1.png").exists())

    def test_invalid_early_stop_rejected_and_curriculum_independent(self):
        ns = self.ns
        before = ns["cm_grid_size"](7000)
        ns["EARLY_STOP_AT"]["CM"] = 7000
        self.assertEqual(before, ns["cm_grid_size"](7000))
        ns["EARLY_STOP_AT"]["FM"] = 0
        with self.assertRaises(ValueError):
            ns["train_method"](
                "FM", ns["fresh_model"](), ns["fm_loss"], ns["sample_fm"]
            )

    def test_native_muon_one_update(self):
        if not hasattr(torch.optim, "Muon"):
            self.skipTest("native Muon is not available in this runtime")
        ns = self.ns
        ns["MUON_BACKEND"] = "native"
        model = ns["fresh_model"]()
        optimizers, info = ns["build_optimizers"](model)
        self.assertEqual(info["backend"], "torch.optim.Muon")
        images, labels = next(iter(ns["train_loader"]))
        loss, _ = ns["fm_loss"](model, images, labels, 1)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_all_cells_in_order_with_synthetic_data(self):
        ns = self.ns
        ns.update(sqrtm=sqrtm, FEATURE_EPOCHS=1, EVAL_N=20, NFE_LIST=[1, 2])
        ns["EARLY_STOP_AT"] = {name: 1 for name in ns["EARLY_STOP_AT"]}
        ns["DEFAULT_NFE"] = {name: 1 for name in ns["DEFAULT_NFE"]}
        images = torch.randn(20, 1, 32, 32).clamp(-1, 1)
        labels = torch.arange(10).repeat_interleave(2)
        dataset = TensorDataset(images, labels)
        ns["train_loader"] = DataLoader(
            dataset, batch_size=4, shuffle=True, generator=ns["loader_generator"]
        )
        ns["test_loader"] = DataLoader(dataset, batch_size=4)
        with patch("builtins.print"), patch.object(plt, "show"):
            for cell in NOTEBOOK["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tag = cell["metadata"]["tags"][0]
                if tag in ("setup", "data"):
                    continue
                with self.subTest(cell=tag):
                    exec(compile("".join(cell["source"]), tag, "exec"), ns)
        self.assertEqual(set(ns["RESULTS"]), set(ns["EARLY_STOP_AT"]))
        self.assertTrue(
            all(info["train_steps"] == 1 for info in ns["RUN_INFO"].values())
        )
        self.assertTrue((Path(self.folder.name) / "summary.csv").exists())
        self.assertEqual(len(list(Path(self.folder.name).glob("*/last.pt"))), 7)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
