import json
from pathlib import Path

path = Path('mean_flow_experiment1.ipynb')
nb = json.loads(path.read_text())


def src(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


def code(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': text.strip() + '\n',
    }


def split_between(text, start, end=None):
    i = text.index(start)
    if end is None:
        return text[i:]
    j = text.index(end, i)
    return text[i:j]


new_cells = []
for cell in nb['cells']:
    text = src(cell)

    if cell.get('cell_type') == 'code' and 'class ScalarEmbed' in text and 'class TinyDiT' in text:
        scalar = split_between(text, 'class ScalarEmbed', '\n\nclass DiTBlock')
        block = split_between(text, 'class DiTBlock', '\n\nclass TinyDiT')
        model = split_between(text, 'class TinyDiT', '\n\ndef fresh_model')
        init = split_between(text, 'def fresh_model')

        new_cells.append(md('### 0-3-1. Scalar time embedding\n\n현재 시간 \\(t\\)와 interval \\(h=t-r\\)를 각각 임베딩할 때 사용하는 공통 Fourier/MLP 모듈이다.\n'))
        new_cells.append(code(scalar))
        new_cells.append(md('### 0-3-2. DiT block — adaLN-Zero\n\n각 Transformer block의 attention/MLP와 conditioning modulation을 정의한다. modulation 마지막 선형층은 zero initialization으로 시작한다.\n'))
        new_cells.append(code(block))
        new_cells.append(md('### 0-3-3. Conditional TinyDiT\n\n입력 patch embedding, positional embedding, \\(t\\)/\\(h\\)/class conditioning, final adaLN modulation과 image projection을 한 곳에 정의한다.\n'))
        new_cells.append(code(model))
        new_cells.append(md('### 0-3-4. 모델 생성 및 parameter 수 확인\n'))
        new_cells.append(code(init))
        continue

    if cell.get('cell_type') == 'code' and 'def logit_normal_times' in text and 'def meanflow_loss' in text:
        timing = split_between(text, 'def logit_normal_times', '\n\ndef linear_path')
        path_helpers = split_between(text, 'def linear_path', '\n\ndef maybe_drop_labels')
        labels = split_between(text, 'def maybe_drop_labels', '\n\ndef meanflow_outputs')
        outputs = split_between(text, 'def meanflow_outputs', '\n\ndef meanflow_loss')
        loss = split_between(text, 'def meanflow_loss')

        new_cells.append(md('### 0-4-1. MeanFlow time sampler\n\n`LogitNormal(-0.4, 1.0)`에서 두 시간을 뽑아 정렬하고, batch의 75%를 diagonal \\(r=t\\) sample로 만든다.\n'))
        new_cells.append(code(timing))
        new_cells.append(md('### 0-4-2. Linear path와 conditional velocity\n'))
        new_cells.append(code(path_helpers))
        new_cells.append(md('### 0-4-3. Class condition dropout\n\n학습 중 일부 label을 null class로 바꿔 기존 label-less 진단 호출과도 호환되게 한다.\n'))
        new_cells.append(code(labels))
        new_cells.append(md('### 0-4-4. MeanFlow JVP target\n\nJVP로 total derivative를 계산하고 stop-gradient target을 만든다.\n'))
        new_cells.append(code(outputs))
        new_cells.append(md('### 0-4-5. Adaptive MeanFlow loss\n\n각 sample의 raw SSE에 adaptive normalization을 적용한다. raw SSE와 interval/diagonal 비율은 별도로 기록한다.\n'))
        new_cells.append(code(loss))
        continue

    if cell.get('cell_type') == 'code' and 'def sample_meanflow_one_step' in text:
        # Keep the verification part before the sampler separate from inference visualization.
        marker = '@torch.no_grad()\ndef sample_meanflow_one_step'
        if marker in text:
            before, after = text.split(marker, 1)
            if before.strip():
                new_cells.append(md('### 0-6-1. Diagonal velocity / denoiser / score sanity check\n'))
                new_cells.append(code(before))
            sampler_body = marker + after
            new_cells.append(md('### 0-6-2. Conditional one-step sampling\n\nFashionMNIST class 0–9를 반복해서 조건으로 넣고 one-step 결과를 확인한다.\n'))
            new_cells.append(code(sampler_body))
            continue

    new_cells.append(cell)

nb['cells'] = new_cells

# Remove stale execution products everywhere and normalize source representation.
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    if isinstance(cell.get('source'), list):
        cell['source'] = ''.join(cell['source'])

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print('reformatted', path, 'cells:', len(nb['cells']))
