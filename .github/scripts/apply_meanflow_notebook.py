import json
from pathlib import Path

PATH = Path('mean_flow_sanity_check.ipynb')
nb = json.loads(PATH.read_text(encoding='utf-8'))

def lines(s):
    return s.splitlines(keepends=True)

def txt(c):
    return ''.join(c.get('source', []))

meta = nb.setdefault('metadata', {})
meta['accelerator'] = 'GPU'
meta.setdefault('colab', {})['gpuType'] = 'T4'
meta['colab'].setdefault('provenance', [])
meta['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
meta['language_info'] = {'name': 'python', 'version': '3.x'}

setup = '''# @title 0-1. Install / imports / configuration
!pip -q install datasets tensorboard

import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.func import jvp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not connected. In Colab choose Runtime > Change runtime type > T4 GPU, reconnect, then run from the top."
    )

DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
print("GPU:", GPU_NAME)
print("CUDA available:", torch.cuda.is_available())

if "T4" not in GPU_NAME:
    warnings.warn(
        f"This notebook is sized for a T4, but current GPU is {GPU_NAME}."
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.mha.set_fastpath_enabled(False)

TRAIN_STEPS = 20_000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8

SCALAR_LOG_EVERY = 50
DIAGNOSTIC_EVERY = 250
SAMPLE_EVERY = 1_000
DIAGNOSTIC_BATCH_SIZE = 64
FIXED_SAMPLE_COUNT = 16
SAMPLE_UPSCALE = 6

P_MEAN = -0.4
P_STD = 1.0
DATA_PROPORTION = 0.75
NORM_P = 1.0
NORM_EPS = 1.0

ROOT_DIR = "/content/meanflow_mnist_dit_sanity"
RUN_NAME = "mnist_dit20k_" + time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(ROOT_DIR, RUN_NAME)
SAMPLE_DIR = os.path.join(RUN_DIR, "samples")
TENSORBOARD_ROOT = os.path.join(ROOT_DIR, "tensorboard")
LOG_DIR = os.path.join(TENSORBOARD_ROOT, RUN_NAME)
REFERENCE_PATH = os.path.join(RUN_DIR, "reference_mnist.png")
CHECKPOINT_PATH = os.path.join(RUN_DIR, "meanflow_dit_mnist_20k.pt")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)
writer.add_text(
    "run/config",
    (
        f"gpu={GPU_NAME}, steps={TRAIN_STEPS}, batch={BATCH_SIZE}, "
        f"lr={LEARNING_RATE}, p_mean={P_MEAN}, p_std={P_STD}, "
        f"data_proportion={DATA_PROPORTION}, norm_eps={NORM_EPS}"
    ),
    global_step=0,
)
writer.flush()

print("RUN_DIR:", RUN_DIR)
print("SAMPLE_DIR:", SAMPLE_DIR)
print("TensorBoard log:", LOG_DIR)
'''

tb_md = '''## 1. TensorBoard — 학습 전에 실행

메트릭은 PNG 파일을 계속 덮어쓰지 않고 **TensorBoard event log**에 누적한다.

- `train/loss_adaptive`
- `train/grad_norm`
- `diagnostic/raw_mse_all`
- `diagnostic/raw_mse_interval`
- `diagnostic/interval_cosine`
- `diagnostic/boundary_mse`
- `run/elapsed_minutes`

생성 이미지만 `samples/step_XXXXX.png`로 1000 step마다 각각 별도 저장한다.
'''

tb_code = '''# @title 1-1. Launch TensorBoard before training
%load_ext tensorboard
%tensorboard --logdir /content/meanflow_mnist_dit_sanity/tensorboard --reload_interval 5
'''

# Keep existing fixed diagnostic definitions/sample saver, but drop CSV and summary-PNG writers.
diag = next(c for c in nb['cells'] if '# @title 4-1. Fixed batch / diagnostics / file writers' in txt(c))
diag_text = txt(diag)
diag_text = diag_text.replace('metric_rows = []\n\n\n', '')
cut = diag_text.find('\ndef write_metrics_csv():')
if cut != -1:
    tail = diag_text.find('\nprint(\n    "fixed finite-interval samples:"', cut)
    diag_text = diag_text[:cut] + diag_text[tail:]
diag_text = diag_text.replace('# @title 4-1.', '# @title 5-1.', 1)
diag['source'] = lines(diag_text)

train = next(c for c in nb['cells'] if '# @title 5-1. Train 20,000 steps' in txt(c))
train_text = txt(train)
train_text = train_text.replace('# @title 5-1.', '# @title 6-1.', 1)
old_zero = '''metric_rows.append(
    {
        "step": 0,
        **step_zero_diagnostics,
        "grad_norm": 0.0,
        "elapsed_minutes": 0.0,
    }
)

write_metrics_csv()
write_training_summary()
'''
new_zero = '''for metric_name, metric_value in step_zero_diagnostics.items():
    writer.add_scalar(
        f"diagnostic/{metric_name}",
        metric_value,
        0,
    )
writer.add_scalar("train/grad_norm", 0.0, 0)
writer.add_scalar("run/elapsed_minutes", 0.0, 0)
writer.flush()
'''
train_text = train_text.replace(old_zero, new_zero)
old_diag = '''        metric_rows.append(
            {
                "step": step,
                **diagnostics,
                "grad_norm": grad_norm,
                "elapsed_minutes": (
                    elapsed_minutes
                ),
            }
        )

        write_metrics_csv()
        write_training_summary()
'''
new_diag = '''        for metric_name, metric_value in diagnostics.items():
            writer.add_scalar(
                f"diagnostic/{metric_name}",
                metric_value,
                step,
            )
        writer.add_scalar(
            "run/elapsed_minutes",
            elapsed_minutes,
            step,
        )
        writer.flush()
'''
train_text = train_text.replace(old_diag, new_diag)
needle = '    optimizer.step()\n\n    if step % DIAGNOSTIC_EVERY == 0:\n'
insert = '''    optimizer.step()

    if step == 1 or step % SCALAR_LOG_EVERY == 0:
        writer.add_scalar(
            "train/loss_adaptive",
            loss.item(),
            step,
        )
        writer.add_scalar(
            "train/grad_norm",
            grad_norm,
            step,
        )

    if step % DIAGNOSTIC_EVERY == 0:
'''
train_text = train_text.replace(needle, insert)
train_text = train_text.replace(
    'print("training complete")\nprint("checkpoint:", CHECKPOINT_PATH)\n',
    'writer.flush()\nwriter.close()\n\nprint("training complete")\nprint("checkpoint:", CHECKPOINT_PATH)\n'
)
train['source'] = lines(train_text)

# Replace section markdown and result cell.
for c in nb['cells']:
    s = txt(c)
    if s.startswith('## 1. MNIST'):
        c['source'] = lines(s.replace('## 1. MNIST', '## 2. MNIST', 1))
    elif '# @title 1-1. Download MNIST' in s:
        c['source'] = lines(s.replace('# @title 1-1.', '# @title 2-1.', 1))
    elif s.startswith('## 2. DiT backbone'):
        c['source'] = lines(s.replace('## 2. DiT backbone', '## 3. DiT backbone', 1))
    elif '# @title 2-1. ~3.78M MeanFlow DiT' in s:
        c['source'] = lines(s.replace('# @title 2-1.', '# @title 3-1.', 1))
    elif s.startswith('## 3. MeanFlow objective'):
        c['source'] = lines(s.replace('## 3. MeanFlow objective', '## 4. MeanFlow objective', 1))
    elif '# @title 3-1. MeanFlow sampling' in s:
        c['source'] = lines(s.replace('# @title 3-1.', '# @title 4-1.', 1))
    elif s.startswith('## 4. Fixed diagnostics'):
        c['source'] = lines('''## 5. Fixed diagnostics and separate sample files

고정된 test image/noise/time pair로 250 step마다 진단하고 **TensorBoard에 누적**한다.
생성 이미지는 1000 step마다 `samples/step_XXXXX.png`로 각각 독립 저장한다.
''')
    elif s.startswith('## 5. Train 20k'):
        c['source'] = lines('''## 6. Train 20k

- adaptive loss / grad norm은 TensorBoard에 50 step마다 기록
- fixed diagnostics는 TensorBoard에 250 step마다 기록
- 생성은 1000 step마다 **별도 PNG** 저장
- 마지막 checkpoint 저장
''')
    elif s.startswith('## 6. Result paths'):
        c['source'] = lines('''## 7. Result paths

생성 결과는 **시점별 독립 PNG**이고, 메트릭은 TensorBoard event log에 있다.
''')
    elif '# @title 6-1. Print saved files and final metrics' in s:
        c['source'] = lines('''# @title 7-1. Print result paths
sample_files = sorted(
    file_name
    for file_name in os.listdir(SAMPLE_DIR)
    if file_name.endswith(".png")
)

print("sample files:")
for file_name in sample_files:
    print(" -", os.path.join(SAMPLE_DIR, file_name))

print("TensorBoard root:", TENSORBOARD_ROOT)
print("reference:", REFERENCE_PATH)
print("checkpoint:", CHECKPOINT_PATH)
''')
    elif '# @title 0-1. Install / imports / configuration' in s:
        c['source'] = lines(setup)

# Insert TensorBoard cells after setup, removing any old copy first.
nb['cells'] = [c for c in nb['cells'] if 'Launch TensorBoard before training' not in txt(c) and not txt(c).startswith('## 1. TensorBoard')]
setup_idx = next(i for i,c in enumerate(nb['cells']) if '# @title 0-1.' in txt(c))
nb['cells'][setup_idx+1:setup_idx+1] = [
    {'cell_type':'markdown','metadata':{},'source':lines(tb_md)},
    {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':lines(tb_code)},
]

for c in nb['cells']:
    s = txt(c)
    if s.startswith('[![Open In Colab]'):
        s = s.replace('수치 결과는 `metrics.csv`와 `training_summary.png`에 저장한다.', '수치 메트릭은 TensorBoard event log에 누적하고, 생성 이미지는 step별 PNG로 각각 저장한다.')
        c['source'] = lines(s)
        break

serialized = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
json.loads(serialized)
PATH.write_text(serialized, encoding='utf-8')
print('updated', PATH)
