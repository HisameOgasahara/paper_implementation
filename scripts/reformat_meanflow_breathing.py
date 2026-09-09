import json
from pathlib import Path

NOTEBOOK = Path('mean_flow_experiment1.ipynb')
nb = json.loads(NOTEBOOK.read_text())


def text(cell):
    src = cell.get('source', '')
    return ''.join(src) if isinstance(src, list) else src


def code_cell(src):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': src.strip() + '\n',
    }


def markdown_cell(src):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': src.strip() + '\n',
    }


new_cells = []

for cell in nb['cells']:
    src = text(cell)

    # Boundary/JVP verification: split preparation, identity check, and derived quantities.
    if (
        cell.get('cell_type') == 'code'
        and "boundary_target_error =" in src
        and "score_from_meanflow" in src
    ):
        new_cells.extend([
            markdown_cell('### 0-7-1. Boundary 검증 입력 준비\n\n작은 batch를 골라 diagonal slice \\(r=t\\)에서 MeanFlow identity를 확인한다.'),
            code_cell('''clean_images, _ = next(iter(train_loader))
clean_images = clean_images[:16].to(DEVICE)

# Avoid t=0 because the score conversion divides by t.
t = torch.rand(clean_images.size(0), device=DEVICE).clamp_min(0.05)
noise = torch.randn_like(clean_images)

z_t = linear_path(clean_images, noise, t)
velocity = conditional_velocity(clean_images, noise)

zero = torch.zeros_like(t)
one = torch.ones_like(t)'''),
            markdown_cell('### 0-7-2. \\(r=t\\)에서 MeanFlow target 확인\n\nDiagonal에서는 \\(t-r=0\\)이므로 target이 instantaneous velocity와 같아야 한다.'),
            code_cell('''with torch.enable_grad():
    average_velocity, total_derivative = jvp(
        lambda z_arg, r_arg, t_arg: meanflow_ema(
            z_arg,
            r_arg,
            t_arg,
        ),
        (z_t, t, t),
        (velocity, zero, one),
    )

boundary_target = (
    velocity
    - (t - t)[:, None, None, None] * total_derivative
)

boundary_target_error = (
    boundary_target - velocity
).abs().max().item()

print('r=t target-v max error:', boundary_target_error)'''),
            markdown_cell('### 0-7-3. Velocity · denoiser · score 변환 확인\n\n같은 diagonal slice에서 뒤 분석에 사용할 세 표현의 shape를 확인한다.'),
            code_cell('''with torch.no_grad():
    velocity_model = instantaneous_velocity(
        meanflow_ema,
        z_t,
        t,
    )
    denoised = denoiser_from_meanflow(
        meanflow_ema,
        z_t,
        t,
    )
    score = score_from_meanflow(
        meanflow_ema,
        z_t,
        t,
    )

print('velocity shape:', tuple(velocity_model.shape))
print('denoiser shape:', tuple(denoised.shape))
print('score shape:', tuple(score.shape))'''),
        ])
        continue

    # One-step sampler: function definition and preview should be separate cells.
    if (
        cell.get('cell_type') == 'code'
        and 'def sample_meanflow_one_step' in src
    ):
        new_cells.extend([
            markdown_cell('### 0-8-1. Conditional 1-step sampler\n\nNoise \\(z_1\\)에서 한 번의 MeanFlow update로 FashionMNIST sample을 생성한다.'),
            code_cell('''@torch.no_grad()
def sample_meanflow_one_step(
    model,
    sample_count=64,
    seed=123,
    labels=None,
):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    z_1 = torch.randn(
        sample_count,
        1,
        28,
        28,
        generator=generator,
        device=DEVICE,
    )

    t = torch.ones(sample_count, device=DEVICE)
    r = torch.zeros(sample_count, device=DEVICE)

    if labels is None:
        labels = (
            torch.arange(sample_count, device=DEVICE)
            % NUM_CLASSES
        )

    average_velocity = model(
        z_1,
        r,
        t,
        labels,
    )

    return z_1 - average_velocity'''),
            markdown_cell('### 0-8-2. 1-step 생성 결과 미리보기\n\n라벨 0–9를 반복해서 조건으로 주고 생성 결과를 grid로 확인한다.'),
            code_cell('''generated_preview = sample_meanflow_one_step(
    meanflow_ema,
    sample_count=64,
)

grid = make_grid(
    generated_preview[:64].cpu(),
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
)

plt.figure(figsize=(8, 8))
plt.imshow(
    grid.permute(1, 2, 0).squeeze(),
    cmap='gray',
)
plt.axis('off')
plt.title(
    'Conditional MeanFlow 1-step FashionMNIST samples '
    '(labels 0-9 repeated)'
)
plt.show()'''),
        ])
        continue

    # Training log plot: keep it away from sampling cell and expand arguments.
    if (
        cell.get('cell_type') == 'code'
        and "training_log.plot" in src
    ):
        new_cells.extend([
            markdown_cell('### 0-8-3. Training curve\n\n기준 모델의 loss와 identity residual이 학습 동안 어떻게 변했는지 확인한다.'),
            code_cell('''training_log.plot(
    x='step',
    y=['loss', 'identity_residual'],
    figsize=(8, 4),
)

plt.show()'''),
        ])
        continue

    # Data cell: expand the most visibly compressed constructor calls.
    if (
        cell.get('cell_type') == 'code'
        and 'def collate_fashion_mnist' in src
        and 'train_loader = DataLoader' in src
    ):
        new_cells.extend([
            code_cell('''try:
    dataset = load_dataset('zalando-datasets/fashion_mnist')
except Exception:
    dataset = load_dataset('anonyme449/fashion_mnist')


to_tensor = ToTensor()


def collate_fashion_mnist(batch):
    images = torch.stack([
        to_tensor(item['image'])
        for item in batch
    ])
    images = images * 2.0 - 1.0

    labels = torch.tensor(
        [item['label'] for item in batch],
        dtype=torch.long,
    )

    return images, labels'''),
            markdown_cell('### 0-2-1. DataLoader 구성'),
            code_cell('''train_loader = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fashion_mnist,
)

test_loader = DataLoader(
    dataset['test'],
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fashion_mnist,
)

sample_images, sample_labels = next(iter(train_loader))

print(
    sample_images.shape,
    sample_images.min().item(),
    sample_images.max().item(),
)
print(sample_labels.shape)'''),
        ])
        continue

    new_cells.append(cell)

nb['cells'] = new_cells

# Clear all stale outputs after structural edits.
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

NOTEBOOK.write_text(
    json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
)

print('reformatted:', NOTEBOOK)
print('cells:', len(nb['cells']))
