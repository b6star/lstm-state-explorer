# LSTM State Explorer

문장을 한 단어씩 읽을 때 **LSTM의 128차원 hidden state와 cell state가 어떻게 업데이트되는지** 관찰하는 Python 실습입니다.

> 직접 구현하기 위한 골격 버전입니다. Python 파일에는 import와 함수 선언만 있으며, 함수 본문은 모두 `pass`입니다. 학습·출력·저장·시각화는 아직 동작하지 않습니다.

## 실습 내용

IMDB 영화 리뷰로 긍정·부정 분류 모델을 학습하고, 짧은 문장을 입력해 단어별 상태 변화를 확인하는 것이 목표입니다.

```text
현재 단어 + 이전 h, c → LSTM → 새로운 h, c
마지막 hidden state → 분류기 → 긍정 점수
```

**단어 하나를 처리할 때마다 h의 128개 값과 c의 128개 값을 각각 한 줄씩 출력**합니다. 한 단계에서 총 256개의 상태 값을 확인하며, 각 줄에 다음 정보를 표시하도록 구현합니다.

```text
상태·차원     이전 값     현재 값     변화량
h[000]         ...         ...         ...
h[001]         ...         ...         ...
```

위 예시는 형식만 보여 줍니다. 실제 구현에서는 h와 c 모두 **0번부터 127번까지 생략 없이 출력**하고, 변화량은 `현재 값 - 이전 값`으로 계산합니다. 결과를 저장하고 히트맵으로 그려 전체 변화 패턴도 살펴봅니다.

## 비교 예시

```text
this movie is good
this movie is not good
```

같은 모델을 사용하고 각 문장의 초기 상태를 0으로 설정합니다. 공통 부분인 `this movie is`까지의 상태가 일치하는지, `not`을 읽으면서 각 차원이 얼마나 바뀌는지 확인합니다. 두 문장의 마지막 `good`을 처리한 상태와 최종 긍정 점수도 비교합니다.

같은 단어라도 이전 기억이 다르면 다른 표현을 만들 수 있습니다. 다만 각 차원에 ‘긍정’·‘부정’이라는 고정된 의미가 있는 것은 아니며, 변화량이 크다고 그 차원이 분류에 더 중요한 것도 아닙니다.

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
├── visualize.py
└── examples/
    └── sentences.txt
```

TensorFlow/Keras, NumPy, Matplotlib을 사용합니다. 구현 후 실행 방법과 실제 관찰 결과를 추가할 예정입니다.
