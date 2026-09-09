import json
from pathlib import Path

path = Path('mean_flow_experiment1.ipynb')
nb = json.loads(path.read_text())


def source(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def set_source(cell, text):
    cell['source'] = text
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []


def find_cell(token, kind=None):
    for cell in nb['cells']:
        if kind is not None and cell.get('cell_type') != kind:
            continue
        if token in source(cell):
            return cell
    raise RuntimeError(f'cell not found: {token}')


setup = find_cell('TRAIN_STEPS = 3000', 'code')
text = source(setup)
text = text.replace('TRAIN_STEPS = 3000', 'TRAIN_STEPS = 20_000')
text = text.replace('LEARNING_RATE = 0.0002', 'LEARNING_RATE = 0.0003')
text = text.replace('SNAPSHOT_EVERY = 500', 'SNAPSHOT_EVERY = 1000')
text = text.replace(
    'BORDER_PROB = 0.25',
    'P_MEAN = -0.4\nP_STD = 1.0\nDATA_PROPORTION = 0.75\nNORM_EPS = 0.01\nNUM_CLASSES = 10\nNULL_CLASS = NUM_CLASSES\nLABEL_DROPOUT = 0.10',
)
set_source(setup, text)


dataset_md = find_cell('라벨은 생성학습에는 사용하지 않고', 'markdown')
set_source(
    dataset_md,
    source(dataset_md).replace(
        '라벨은 생성학습에는 사용하지 않고,\nStanczuk의 class-wise manifold 분석과 Li의 linear probing에서만 사용한다.',
        '라벨은 **MeanFlow 생성학습의 class condition**으로 사용한다. 분석 단계에서는 같은 라벨을 그대로 전달하며,\n라벨이 없는 기존 진단 호출은 classifier-free null condition으로 동작하도록 호환 경로를 둔다.',
    ),
)


model_cell = find_cell('class TinyDiT', 'code')
model_code = '''class ScalarEmbed(nn.Module):

    def __init__(self, dim=224, fourier=128):
        super().__init__()
        self.fourier = fourier
        self.mlp = nn.Sequential(
            nn.Linear(fourier, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, scalar):
        scalar = scalar.reshape(-1, 1)
        half = self.fourier // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=scalar.device, dtype=scalar.dtype)
            / half
        )
        phase = scalar * frequencies[None] * 2.0 * math.pi
        embedding = torch.cat([phase.cos(), phase.sin()], dim=1)
        return self.mlp(embedding)


class DiTBlock(nn.Module):

    def __init__(self, dim=224, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens, conditioning):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(conditioning).chunk(6, dim=-1)
        hidden = self.norm1(tokens)
        hidden = hidden * (1.0 + scale1[:, None]) + shift1[:, None]
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=True)
        tokens = tokens + gate1[:, None] * attended
        hidden = self.norm2(tokens)
        hidden = hidden * (1.0 + scale2[:, None]) + shift2[:, None]
        tokens = tokens + gate2[:, None] * self.feed_forward(hidden)
        return tokens


class TinyDiT(nn.Module):

    def __init__(self, dim=224, depth=4, heads=8, patch=4):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.input_projection = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.position = nn.Parameter(torch.zeros(1, 49, dim))
        self.t_embedding = ScalarEmbed(dim)
        self.h_embedding = ScalarEmbed(dim)
        self.class_embedding = nn.Embedding(NUM_CLASSES + 1, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.output_projection = nn.Linear(dim, patch * patch)

        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _labels(self, batch_size, device, labels):
        if labels is None:
            return torch.full((batch_size,), NULL_CLASS, device=device, dtype=torch.long)
        return labels.to(device=device, dtype=torch.long)

    def _conditioning(self, r, t, labels):
        h = t - r
        labels = self._labels(t.shape[0], t.device, labels)
        return self.t_embedding(t) + self.h_embedding(h) + self.class_embedding(labels)

    def _tokens(self, images):
        tokens = self.input_projection(images)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens + self.position

    def forward_features(self, images, r, t, labels=None):
        tokens = self._tokens(images)
        conditioning = self._conditioning(r, t, labels)
        pooled_features = []
        for block in self.blocks:
            tokens = block(tokens, conditioning)
            pooled_features.append(tokens.mean(dim=1))
        return pooled_features

    def forward(self, images, r, t, labels=None):
        tokens = self._tokens(images)
        conditioning = self._conditioning(r, t, labels)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        shift, scale = self.final_modulation(conditioning).chunk(2, dim=-1)
        tokens = self.final_norm(tokens)
        tokens = tokens * (1.0 + scale[:, None]) + shift[:, None]
        patches = self.output_projection(tokens)
        patches = patches.view(images.size(0), 7, 7, self.patch, self.patch)
        images_out = patches.permute(0, 1, 3, 2, 4)
        return images_out.reshape(images.size(0), 1, 28, 28)


def fresh_model():
    return TinyDiT().to(DEVICE)


parameter_count = sum(parameter.numel() for parameter in fresh_model().parameters())
print('parameters:', parameter_count)
'''
set_source(model_cell, model_code)


model_md = find_cell('## 0-3. 공통 TinyDiT', 'markdown')
set_source(
    model_md,
    '''## 0-3. Conditional TinyDiT — 공식 MeanFlow/DiT 설정에 맞춘 기준 backbone

기존 `dim=96, depth=3` 및 공통 scalar embedding 대신 다음을 사용한다.

- patch size \(4\), \(7\times7=49\) tokens
- hidden dimension \(224\), depth \(4\), attention heads \(8\)
- 시간은 \(t\)와 interval \(h=t-r\)를 별도 embedding
- FashionMNIST class \(y\in\{0,\dots,9\}\)를 class embedding으로 조건화
- 기존 분석 호출과 호환하기 위한 null class 추가
- 각 DiT block modulation과 final modulation/output을 zero initialization (adaLN-Zero)

`forward_features`는 뒤의 representation 분석을 위해 유지한다. JVP 호환성을 위해 attention은 `need_weights=True` 경로를 사용한다.
''',
)


objective_cell = find_cell('def sample_meanflow_times', 'code')
objective_code = '''def logit_normal_times(batch_size, device):
    normal = torch.randn(batch_size, device=device) * P_STD + P_MEAN
    return torch.sigmoid(normal)


def sample_meanflow_times(batch_size, device):
    first = logit_normal_times(batch_size, device)
    second = logit_normal_times(batch_size, device)
    t = torch.maximum(first, second)
    r = torch.minimum(first, second)

    diagonal_count = int(batch_size * DATA_PROPORTION)
    if diagonal_count > 0:
        indices = torch.randperm(batch_size, device=device)[:diagonal_count]
        r = r.clone()
        r[indices] = t[indices]
    return r, t


def linear_path(clean_images, noise, t):
    scale = t[:, None, None, None]
    return (1.0 - scale) * clean_images + scale * noise


def conditional_velocity(clean_images, noise):
    return noise - clean_images


def maybe_drop_labels(labels):
    if labels is None:
        return None
    labels = labels.clone()
    drop_mask = torch.rand(labels.shape[0], device=labels.device) < LABEL_DROPOUT
    labels[drop_mask] = NULL_CLASS
    return labels


def meanflow_outputs(model, clean_images, labels=None):
    batch_size = clean_images.size(0)
    r, t = sample_meanflow_times(batch_size, clean_images.device)
    noise = torch.randn_like(clean_images)
    z_t = linear_path(clean_images, noise, t)
    velocity = conditional_velocity(clean_images, noise)
    labels_for_model = maybe_drop_labels(labels)

    zero = torch.zeros_like(t)
    one = torch.ones_like(t)

    def model_function(z_value, t_value, r_value):
        return model(z_value, r_value, t_value, labels_for_model)

    average_velocity, total_derivative = jvp(
        model_function,
        (z_t, t, r),
        (velocity, one, zero),
    )
    target = (
        velocity
        - (t - r)[:, None, None, None] * total_derivative
    ).detach()
    return average_velocity, target, r, t


def meanflow_loss(model, clean_images, labels=None):
    prediction, target, r, t = meanflow_outputs(model, clean_images, labels)
    squared_error = (prediction - target).pow(2).flatten(1).sum(dim=1)
    adaptive_weight = (squared_error.detach() + NORM_EPS).reciprocal()
    loss = (squared_error * adaptive_weight).mean()

    auxiliary = {
        'raw_sse': squared_error.mean().detach(),
        'interval_fraction': (r < t).float().mean().detach(),
        'diagonal_fraction': (r == t).float().mean().detach(),
    }
    return loss, auxiliary
'''
set_source(objective_cell, objective_code)


objective_md = find_cell('## 0-4. MeanFlow 학습식', 'markdown')
set_source(
    objective_md,
    '''## 0-4. MeanFlow objective — 공식 설정을 소형 FashionMNIST에 맞게 복구

MeanFlow identity와 linear Gaussian path 자체는 기존 노트북과 동일하게 유지한다.

\[
u(z_t,r,t)=v(z_t,t)-(t-r)\frac{d}{dt}u(z_t,r,t)
\]

JVP tangent도 기존대로 \((v,0,1)\)이다.

이번 수정에서 바뀌는 학습 설정은 다음과 같다.

- \(t,r\) 후보를 `LogitNormal(-0.4, 1.0)`에서 뽑아 정렬
- batch의 75%는 \(r=t\) diagonal sample, 25%는 interval sample
- raw SSE 대신 `NORM_EPS=0.01` adaptive normalization
- FashionMNIST label을 class condition으로 전달
- 10% label dropout으로 null condition도 함께 학습해 기존 label-less 진단 호출과 호환
''',
)


train_cell = find_cell('def train_meanflow', 'code')
train_text = source(train_cell)
train_text = train_text.replace('clean_images, _ = next(batch_iterator)', 'clean_images, labels = next(batch_iterator)')
train_text = train_text.replace(
    'clean_images = clean_images.to(DEVICE, non_blocking=True)',
    'clean_images = clean_images.to(DEVICE, non_blocking=True)\n        labels = labels.to(DEVICE, non_blocking=True)',
)
train_text = train_text.replace(
    'loss, auxiliary = meanflow_loss(model, clean_images)',
    'loss, auxiliary = meanflow_loss(model, clean_images, labels)',
)
set_source(train_cell, train_text)


sample_cell = find_cell('def sample_meanflow_one_step', 'code')
sample_text = source(sample_cell)
sample_text = sample_text.replace(
    'def sample_meanflow_one_step(model, sample_count=64, seed=123):',
    'def sample_meanflow_one_step(model, sample_count=64, seed=123, labels=None):',
)
sample_text = sample_text.replace(
    'average_velocity = model(z_1, r, t)',
    'if labels is None:\n        labels = torch.arange(sample_count, device=DEVICE) % NUM_CLASSES\n    average_velocity = model(z_1, r, t, labels)',
)
sample_text = sample_text.replace(
    "plt.title('MeanFlow 1-step FashionMNIST samples')",
    "plt.title('Conditional MeanFlow 1-step FashionMNIST samples (labels 0-9 repeated)')",
)
set_source(sample_cell, sample_text)


for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []


note = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': '> **2026-09-09 기준 모델 수정.** 기존 Stanczuk/Ventura/Niedoba/Li/Qian/Buchanan/AlphaFlow 진단 섹션은 유지하고, 기준 MeanFlow 학습부만 conditional DiT + separate t/h embedding + adaLN-Zero + logit-normal time sampling + adaptive loss normalization으로 교체했다.\n',
}
if not any('2026-09-09 기준 모델 수정' in source(cell) for cell in nb['cells']):
    nb['cells'].insert(1, note)


path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print('patched', path)
