# Flow 모델 구현 실습

[flow_basic.ipynb](flow_basic.ipynb)는 FashionMNIST에서 FM → Rectified Flow/Reflow → CM → CTM → Shortcut → MeanFlow를 학습하고 조건부 생성 결과를 비교한다.

## 실행

[Colab에서 열기](https://colab.research.google.com/github/HisameOgasahara/paper_implementation/blob/main/flow_basic.ipynb) 후 GPU 런타임에서 위에서부터 실행한다. 공통 DiT는 `meanflow_minimal`의 조건부 구조를 사용하며, 입력은 32×32로 padding한 FashionMNIST다. 생성 모델은 Muon+AdamW, 평가용 분류기는 Adam을 사용한다.

`TRAIN_STEPS=20_000`을 유지하면서 `EARLY_STOP_AT`으로 모델별 중단 시점을 지정한다. FM, RF1, RF2, CM, CTM, Shortcut, MeanFlow 모두 기본적으로 각각 5,000 step에서 업데이트와 체크포인트 저장을 마친 뒤 중단한다. RF1과 RF2는 각 단계에서 5,000 step씩 학습한다. 실습 시간 제한을 위한 기본값이며 수렴을 보장하는 임계값은 아니다. 학습률은 상수이며 CM의 격자 curriculum은 20,000-step 기준을 유지한다.

각 모델 절의 결과 셀은 클래스별 생성 그림을 표시한다. 이어지는 직접 생성 셀에서 `requested_labels`, `requested_nfe`, `requested_seed`를 바꿔 원하는 클래스를 생성할 수 있다. RF 절의 직접 생성 셀은 2-RF를 사용한다.

## 모델별 학습

| 방법 | 학습 내용 |
| --- | --- |
| FM | 클래스 조건부 순간 속도 회귀 |
| RF | 독립 1-RF 학습 후, 클래스 라벨을 보존한 teacher coupling으로 reflow |
| CM | additive noise·인접 격자·EMA target을 사용하는 standalone consistency training |
| CTM | FM teacher와 EMA student의 soft consistency 및 대각선 denoising |
| Shortcut | FM target과 두 작은 이동으로 구성한 bootstrap target |
| MeanFlow | minimal의 logit-normal pair·75% 대각선 표본·adaptive loss·JVP |

CM/CTM을 포함한 각 방법의 실습 적응 사항은 노트북에 표시되어 있다. 원 논문의 대규모 benchmark 재현은 아니다. CM의 noise scale과 다른 모델의 선형 시간은 구분한다.

## 결과와 저장

실행마다 별도 `flow_basic_runs` 폴더가 만들어진다. 모델별 `last.pt`에는 online 모델, optimizer 두 개, 해당하는 EMA target, 설정, 실제 step과 중단 이유가 저장된다. `training.csv`와 조건부 생성 PNG도 같은 모델 폴더에 저장한다. 데이터 iterator까지 복원하는 완전 동일한 학습 재개 기능은 포함하지 않는다.

평가에서는 균등 클래스·고정 noise를 사용한다. 중앙 28×28 영역으로 feature-FID/KID와 클래스 조건 일치율을 계산하며, 실제 학습 step과 teacher를 포함한 비용을 함께 기록한다. 서로 다른 중단 시점의 결과는 동일 학습량 비교가 아니다.

## 기존 MeanFlow 기준 구현

[meanflow_minimal](meanflow_minimal/README.md)의 MNIST/FashionMNIST 노트북과 기존 출력은 유지한다. 논문 원문과 출처 목록은 Git에서 제외하는 `reference` 폴더에 있다.

## 로컬 실행 검증

검증용 환경에 `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `nbformat`이 필요하다.

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

검증은 CPU의 축소 모델과 합성 데이터로 수행한다. notebook 문법, minimal과의 수치 일치, JVP, 조건 전달, 각 목표식의 역전파, optimizer, 조기 종료·저장, 셀 실행 순서를 확인한다. 전체 FashionMNIST 학습의 수렴·생성 품질은 GPU 학습 결과로 별도 확인해야 한다.
