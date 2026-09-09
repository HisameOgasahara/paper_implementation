# 변경 이력

## 2026-09-09 — MeanFlow 학습의 JVP 계산량 축소

### 변경 이유

`mean_flow_experiment1.ipynb`를 코랩 T4에서 학습할 때, PyTorch가 선택한 `_scaled_dot_product_efficient_attention` 커널이 정방향 자동 미분(JVP)을 지원하지 않아 중단됐다. `MultiheadAttention`의 빠른 경로를 끄는 설정만으로는 내부 SDPA 커널 선택까지 제한할 수 없었다.

처음에는 모든 attention 호출에 `SDPBackend.MATH`를 지정해 오류를 해결했다. 이후 학습 배치 전체에 필요하지 않은 JVP까지 계산하는 부분을 줄이고, 기본 수학 연산 방식의 적용 범위를 JVP 구간으로 좁혔다.

### 변경 내용

- `meanflow_outputs`에서 시간·노이즈·조건 레이블 드롭아웃을 한 번 샘플링한 뒤, `r=t`와 `r≠t` 샘플을 나눈다.
- `r=t` 샘플은 일반 순전파로 예측하고, 회귀 타깃은 `v`를 사용한다. 이 샘플도 기존처럼 손실과 파라미터 역전파에 참여한다.
- `r≠t` 샘플에만 JVP를 계산해 타깃 `v − (t−r)·JVP`를 구성한다. 예측값은 JVP 호출에서 함께 얻으므로 별도 순전파를 반복하지 않는다.
- 두 결과를 원래 샘플 순서로 복원한 뒤 기존 배치 전체의 적응형 손실을 계산한다. 타깃의 stop-gradient, 조건 레이블, 시간 분포와 학습 배치 크기를 유지한다.
- `meanflow_jvp`가 JVP 실행 중에만 `MATH` attention을 선택한다. 일반 순전파와 생성은 PyTorch의 기본 SDPA 선택을 사용한다. 경계 검증과 AlphaFlow 분석의 직접 JVP 호출도 이 함수를 사용한다.

현재 `BATCH_SIZE=128`, `DATA_PROPORTION=0.75`에서는 96개가 일반 순전파, 32개가 JVP 경로를 사용한다. 일반적으로 대각 샘플 수는 `int(batch_size * DATA_PROPORTION)`이며, 우연히 시간이 같은 추가 샘플도 JVP에서 제외된다. 경계 검증과 AlphaFlow 분석은 기존 분석 계산 범위를 유지한다.

### 수학적 근거와 적용 조건

MeanFlow 타깃은 `v − (t−r)·JVP`다. `r=t`일 때 JVP의 계수가 0이므로 해당 미분을 생략할 수 있다. 현재 TinyDiT는 배치 샘플끼리 연산을 섞지 않고 attention dropout도 0이므로 배치를 분리해도 수학적으로 같은 예측·타깃·손실·파라미터 기울기를 얻는다. 부동소수점 연산 순서와 SDPA backend 차이로 작은 수치 오차는 발생할 수 있다. 향후 BatchNorm이나 확률적 모델 층을 추가하면 동등성을 다시 확인해야 한다.

### 가속 커널 대안과 T4 제약

MeanFlow는 [2025년 5월 19일 공개](https://arxiv.org/abs/2505.13447)됐으며, JVP를 지원하는 가속 attention 구현도 존재한다.

| 대안 | 적용 시 확인할 사항 |
| --- | --- |
| [NVIDIA RCM의 FlashAttention JVP](https://github.com/NVlabs/rcm/blob/main/rcm/utils/flash_attention_jvp_triton.py) | Triton 기반 JVP 구현이다. 사용 GPU, 입력 크기, 정밀도와 역전파 호환성을 검증한 뒤 통합해야 한다. |
| [jvp_flash_attention](https://github.com/amorehead/jvp_flash_attention) | `torch.func.jvp` 사용 시 `JVPAttn.fwd_dual` 경로가 안내돼 있다. 확인한 구현은 헤드 차원에 16·32·64·128·256을 요구하고 시퀀스 길이에도 제약이 있어, 현재 모델의 49개 토큰·헤드 차원 28을 그대로 넣을 수 없다. 패딩을 적용한다면 마스킹과 attention 스케일을 보존해야 한다. |
| [Decoupled MeanFlow의 FA2·FA3 JVP 경로](https://github.com/kyungmnlee/dmf) | 프로젝트에서 JVP 지원 attention 경로를 제공한다. 안내 환경은 FA2용 Ampere, FA3용 Hopper 계열이다. |

[일반 FlashAttention-2 CUDA 구현](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)은 Ampere·Ada·Hopper를 지원 대상으로 안내한다. T4는 Turing 세대이므로 별도 [flash-attention-turing](https://github.com/ssiu/flash-attention-turing) 구현을 확인해야 하며, Turing 지원 자체가 MeanFlow에 필요한 JVP 지원을 보장하지는 않는다. 위 대안들이 모두 T4에서 불가능하다는 의미는 아니지만, 이번 변경에서는 T4에서 검증하지 못한 외부 커널을 추가하지 않았다.

### 검증과 성능 한계

검증 코드는 `tests/test_mean_flow_experiment1.py`에 있다. 전체 배치에 JVP를 적용하는 기준 구현과 분리 구현을 같은 난수·모델 파라미터로 비교하고, 대각 샘플만 있는 경우·구간 샘플만 있는 경우·혼합 배치 및 조건 레이블 없는 입력을 확인한다. CUDA가 있으면 FP16 혼합 정밀도 학습 검사도 실행한다.

로컬 CPU 검증은 4개 테스트가 통과했고 CUDA 테스트 1개는 GPU 부재로 건너뛰었다. 예측값·타깃·손실·전체 파라미터 기울기와 난수 상태의 동등성, JVP 호출 배치 크기, backend 설정 복원, 미분값의 유한차분 비교 및 학습 업데이트를 확인했다.

JVP 대상 샘플 수가 기본 설정에서 128개에서 32개로 감소한다. 전체 학습의 4배 가속을 뜻하지는 않는다. 일반 순전파·역전파는 여전히 모든 샘플에 필요하고, 배치 분할·인덱싱·GPU 동기화 비용도 추가된다. 특히 토큰 수가 49개로 짧아 전용 attention 커널의 큰 시퀀스 벤치마크 배율을 그대로 적용할 수 없다. 로컬 환경은 CPU 전용이므로 T4의 실제 처리 시간·최대 GPU 메모리와 가속 배율은 미측정이다.
