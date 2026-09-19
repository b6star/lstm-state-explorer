# LSTM State Explorer

IMDB 영화 리뷰로 긍정·부정을 분류하도록 학습한 LSTM(Long Short-Term Memory)의 내부 계산 과정을 숫자로 관찰하는 프로젝트입니다.
문장을 입력하면 모델이 단어를 순서대로 처리할 때마다 f(Forget gate), i(Input gate), g(Candidate cell state), o(Output gate), c(Cell state), h(Hidden state)를 계산합니다. 각 벡터의 128개 차원을 빠짐없이 출력하고, 차원별 이전 값·현재 값·변화량을 비교합니다.
이를 통해 새 단어가 들어왔을 때 이전 기억을 얼마나 유지하고, 새 정보를 얼마나 반영하며, 그 결과 상태가 어떻게 바뀌는지 살펴볼 수 있습니다. 관찰에는 학습이 끝난 모델을 사용하며, 가중치는 변경하지 않습니다.

<br>

## 상태와 계산

모두 128차원 벡터입니다.

| 기호 | Full Name | 역할 |
| --- | --- | --- |
| `f` | Forget gate | 이전 cell state의 각 차원을 얼마나 유지할지 정하는 128차원 벡터 |
| `i` | Input gate | 새 기억 후보의 각 차원을 얼마나 반영할지 정하는 128차원 벡터 |
| `g` | Candidate cell state | 현재 단어 벡터와 이전 hidden state로 계산한 새 기억 후보를 담은 128차원 벡터 |
| `o` | Output gate | tanh로 변환한 cell state의 각 차원을 hidden state에 얼마나 반영할지 정하는 128차원 벡터 |
| `c` | Cell state | 이전 cell state에 `f`를 곱하고, 새 기억 후보 `g`에 `i`를 곱해 같은 차원끼리 더한 128차원 벡터 |
| `h` | Hidden state | 현재 cell state에 tanh를 적용한 뒤 같은 차원의 `o`를 곱한 128차원 벡터. 다음 단어 처리와 마지막 단계의 긍정·부정 분류에 사용 |

원소 범위: `f` 0\~1, `i` 0\~1, `g` -1\~1, `o` 0\~1.

<br>

$$
\begin{aligned}
f &= \sigma(W_f x + U_f h_{\mathrm{prev}} + b_f) \\
i &= \sigma(W_i x + U_i h_{\mathrm{prev}} + b_i) \\
g &= \tanh(W_g x + U_g h_{\mathrm{prev}} + b_g) \\
o &= \sigma(W_o x + U_o h_{\mathrm{prev}} + b_o) \\
c &= f \odot c_{\mathrm{prev}} + i \odot g \\
h &= o \odot \tanh(c)
\end{aligned}
$$

<br>

| 기호 | 설명 |
| --- | --- |
| $x$ | 현재 단어의 임베딩 벡터 |
| $c_{\mathrm{prev}}$, $h_{\mathrm{prev}}$ | 현재 단어를 처리하기 전 cell state와 hidden state |
| $c$, $h$ | 현재 단어를 처리한 후 cell state와 hidden state |
| $W_f, W_i, W_g, W_o$ | 입력 벡터에 곱하는 학습된 가중치 행렬 |
| $U_f, U_i, U_g, U_o$ | 이전 hidden state에 곱하는 학습된 가중치 행렬 |
| $b_f, b_i, b_g, b_o$ | 학습된 편향 벡터 |
| $\sigma$ | sigmoid: 각 원소를 0\~1 사이로 변환 |
| $\tanh$ | 쌍곡탄젠트: 각 원소를 -1\~1 사이로 변환 |
| $\odot$ | 같은 위치의 원소끼리 곱하기 |

$W x$와 $U h_{\mathrm{prev}}$는 행렬·벡터 곱입니다.

<br>

## 사용법

### 1. 준비 및 다운로드

**Python 3.12(Windows에서는 64비트)**를 설치합니다. 현재 의존성은 TensorFlow 2.16 계열에 맞춰져 있습니다. <br>
확인 환경: Windows / Python 3.12.10 / TensorFlow 2.16.2

```bash
git clone https://github.com/b6star/lstm-state-explorer.git
cd lstm-state-explorer
```

### 2. 가상환경 및 라이브러리 설치

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

이후 명령의 `.\.venv\Scripts\python.exe`를 `./.venv/bin/python`으로 바꿉니다. macOS/Linux 실행은 미검증입니다.

### 3. 모델 학습

```powershell
.\.venv\Scripts\python.exe train_model.py
```

IMDB 리뷰로 긍정·부정을 학습하고 `models/lstm.keras`에 저장합니다. 기본 3에포크이며, 최초 데이터 다운로드에는 인터넷 연결이 필요합니다.

모델은 Git에 포함되지 않습니다. 기존 `models/lstm.keras`를 복사했다면 학습을 생략합니다.

### 4. 단어별 상태 출력

```powershell
.\.venv\Scripts\python.exe inspect_states.py --text "this movie is not good"
```

`<START>`부터 단어마다 `f`·`i`·`g`·`o`·`c`·`h`를 각각 **128줄씩** 출력합니다. 사전에 없는 단어는 모두 ID 2로 처리됩니다.

| 저장 파일 | 내용 |
| --- | --- |
| `outputs/states.npz` | 상태 벡터의 숫자 배열 |
| `outputs/states_f.txt` 등 | `f`, `i`, `g`, `o`, `c`, `h`별 토큰 가로 표 |

같은 경로로 실행하면 기존 결과를 덮어씁니다.

<br>

## 결과 읽기

```text
dim        previous       current         delta
f[038]     직전 단계 값     현재 단계 값     현재 - 직전
```

`f[038]`은 이전 `c[038]`을 유지할 비율입니다. 이 행의 `delta`는 `f[038]` 자체의 변화량입니다.

첫 단계에서 `f`·`i`·`g`·`o`의 이전 값과 변화량은 `--`이며, `c`·`h`의 초기값은 0입니다. 차원 번호에 ‘긍정’ 같은 고정된 의미가 지정되어 있지는 않습니다.

같은 모델로 두 문장을 비교하려면:

```powershell
.\.venv\Scripts\python.exe inspect_states.py --text "this movie is good" --output outputs/good.npz
.\.venv\Scripts\python.exe inspect_states.py --text "this movie is not good" --output outputs/not_good.npz
```

<br>

## 파일 구성

```text
lstm-state-explorer/
├── README.md
├── requirements.txt
├── .gitignore
├── train_model.py
├── inspect_states.py
├── model_io.py
├── text_processing.py
├── state_tracker.py
├── state_output.py
└── examples/
    └── sentences.txt
```

참고: [Keras LSTM 구현](https://github.com/keras-team/keras/blob/v3.10.0/keras/src/layers/rnn/lstm.py)
