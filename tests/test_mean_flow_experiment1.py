"""MeanFlow 실험 노트북의 attention 미분 호환성을 검증한다."""

import ast
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel


NOTEBOOK = Path(__file__).resolve().parents[1] / "mean_flow_experiment1.ipynb"


def load_definitions():
    namespace = dict(
        torch=torch, nn=nn, math=math, jvp=jvp,
        SDPBackend=SDPBackend, sdpa_kernel=sdpa_kernel,
        NUM_CLASSES=10, NULL_CLASS=10, P_MEAN=-0.4, P_STD=1.0,
        DATA_PROPORTION=0.75, LABEL_DROPOUT=0.1, NORM_EPS=0.01,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or not (
            "class TinyDiT" in source or "def meanflow_outputs" in source
        ):
            continue
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.ClassDef, ast.FunctionDef)
        )]
        exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    return namespace


def full_batch_outputs(namespace, model, clean_images, labels=None):
    """최적화 전 전체 배치 JVP를 비교 기준으로 유지한다."""
    r, t = namespace["sample_meanflow_times"](len(clean_images), clean_images.device)
    noise = torch.randn_like(clean_images)
    z_t = namespace["linear_path"](clean_images, noise, t)
    velocity = namespace["conditional_velocity"](clean_images, noise)
    labels = namespace["maybe_drop_labels"](labels)
    prediction, derivative = namespace["meanflow_jvp"](
        lambda z, time, start: model(z, start, time, labels),
        (z_t, t, r),
        (velocity, torch.ones_like(t), torch.zeros_like(r)),
    )
    target = (velocity - (t - r)[:, None, None, None] * derivative).detach()
    return prediction, target, r, t


class MeanFlowAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_definitions()

    def setUp(self):
        self.fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(42)

    def tearDown(self):
        torch.backends.mha.set_fastpath_enabled(self.fastpath)

    def make_model(self, device="cpu"):
        model = self.namespace["TinyDiT"](dim=16, depth=1, heads=2).to(device)
        # 영 초기화로 attention 미분 오류가 가려지지 않도록 한다.
        with torch.no_grad():
            model.output_projection.weight.normal_(std=0.03)
            model.blocks[0].modulation[-1].weight.normal_(std=0.03)
        return model

    def test_jvp_matches_finite_difference(self):
        model = self.make_model().double()
        images = torch.randn(2, 1, 28, 28, dtype=torch.float64)
        velocity = torch.randn_like(images)
        r = torch.full((2,), 0.2, dtype=torch.float64)
        t = torch.full((2,), 0.7, dtype=torch.float64)
        labels = torch.tensor([1, 7])
        fn = lambda z, time: model(z, r, time, labels)
        # JVP 구간에서만 MATH를 사용하고 호출자의 backend 설정을 복원한다.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _, derivative = self.namespace["meanflow_jvp"](
                fn, (images, t), (velocity, torch.ones_like(t))
            )
            eps = 1e-5
            with sdpa_kernel(SDPBackend.MATH):
                finite = (fn(images + eps * velocity, t + eps)
                          - fn(images - eps * velocity, t - eps)) / (2 * eps)
            self.assertFalse(torch.backends.cuda.math_sdp_enabled())
        torch.testing.assert_close(derivative, finite, atol=1e-7, rtol=1e-4)
        self.assertGreater(derivative.norm().item(), 0)

    def test_split_matches_full_batch_outputs_loss_and_gradients(self):
        ns = self.namespace
        images = torch.randn(8, 1, 28, 28, dtype=torch.float64)
        sample_times = ns["logit_normal_times"]
        for proportion in (0.0, 0.75, 1.0):
            for labels in (None, torch.arange(8)):
                with self.subTest(proportion=proportion, conditional=labels is not None):
                    reference = self.make_model().double()
                    optimized = copy.deepcopy(reference)
                    with patch.dict(
                        ns, DATA_PROPORTION=proportion,
                        logit_normal_times=lambda size, device: sample_times(size, device).double(),
                    ):
                        torch.manual_seed(123)
                        expected = full_batch_outputs(ns, reference, images, labels)
                        expected_rng = torch.get_rng_state().clone()
                        torch.manual_seed(123)
                        # 같은 backend에서 배치 분할로 발생한 차이를 비교한다.
                        with sdpa_kernel(SDPBackend.MATH):
                            actual = ns["meanflow_outputs"](optimized, images, labels)
                        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                        for old, new in zip(expected, actual):
                            torch.testing.assert_close(old, new, atol=1e-10, rtol=1e-8)
                        self.assertFalse(actual[1].requires_grad)
                        expected_sse = (expected[0] - expected[1]).square().flatten(1).sum(1)
                        expected_loss = (
                            expected_sse / (expected_sse.detach() + ns["NORM_EPS"])
                        ).mean()
                        torch.manual_seed(123)
                        with sdpa_kernel(SDPBackend.MATH):
                            actual_loss, auxiliary = ns["meanflow_loss"](
                                optimized, images, labels
                            )
                        torch.testing.assert_close(expected_loss, actual_loss)
                        torch.testing.assert_close(auxiliary["raw_sse"], expected_sse.mean())
                        expected_loss.backward()
                        actual_loss.backward()
                        for (name, old), new in zip(
                            reference.named_parameters(), optimized.parameters()
                        ):
                            self.assertIsNotNone(new.grad, name)
                            torch.testing.assert_close(
                                old.grad, new.grad, atol=1e-10, rtol=1e-7, msg=name
                            )

    def test_only_interval_samples_use_jvp_and_math_backend(self):
        ns = self.namespace
        model = self.make_model()
        images = torch.randn(8, 1, 28, 28)
        labels = torch.arange(8)
        for proportion, expected_sizes in ((0.0, [8]), (0.75, [2]), (1.0, [])):
            calls = []
            attention_flags = []

            def record_jvp(function, primals, tangents):
                calls.append(len(primals[0]))
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                return jvp(function, primals, tangents)

            def record_attention(module, args):
                attention_flags.append(torch.backends.cuda.flash_sdp_enabled())

            handle = model.blocks[0].attention.register_forward_pre_hook(record_attention)
            try:
                with self.subTest(proportion=proportion), patch.dict(
                    ns, DATA_PROPORTION=proportion, jvp=record_jvp
                ), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    ns["meanflow_outputs"](model, images, labels)
                    self.assertFalse(torch.backends.cuda.math_sdp_enabled())
            finally:
                handle.remove()
            self.assertEqual(calls, expected_sizes)
            expected_flags = ([True] if proportion > 0 else []) + (
                [False] if proportion < 1 else []
            )
            self.assertEqual(attention_flags, expected_flags)

    def check_loss_and_update(self, device, amp=False):
        model = self.make_model(device)
        images = torch.randn(2, 1, 28, 28, device=device)
        labels = torch.tensor([1, 7], device=device)
        optimizer = torch.optim.AdamW(model.parameters())
        before = model.output_projection.weight.detach().clone()
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            loss, _ = self.namespace["meanflow_loss"](model, images, labels)
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        scaler.step(optimizer)
        scaler.update()
        self.assertFalse(torch.equal(before, model.output_projection.weight))
        model.eval().requires_grad_(False)
        with torch.no_grad():
            prediction, target, _, _ = self.namespace["meanflow_outputs"](
                model, images, labels
            )
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(target.requires_grad)

    def test_cpu_training_and_frozen_evaluation(self):
        self.check_loss_and_update("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU가 필요합니다")
    def test_cuda_amp_training_and_frozen_evaluation(self):
        self.check_loss_and_update("cuda", amp=True)


if __name__ == "__main__":
    unittest.main()
