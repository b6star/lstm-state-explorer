import numpy as np
import tensorflow as tf

def calculate_step(cell, word_vector, h, c):
    # Keras의 가중치 묶음 순서는 i, f, g, o이다.
    z = tf.matmul(word_vector, cell.kernel)
    z = z + tf.matmul(h, cell.recurrent_kernel)

    if cell.use_bias:
        z = z + cell.bias

    z_i, z_f, z_g, z_o = tf.split(z, 4, axis=1)

    i = cell.recurrent_activation(z_i)
    f = cell.recurrent_activation(z_f)
    g = cell.activation(z_g)
    o = cell.recurrent_activation(z_o)

    # 남긴 이전 기억 + 추가할 새 기억
    c = f * c + i * g
    h = o * cell.activation(c)

    return {"f": f, "i": i, "g": g, "o": o, "h": h, "c": c}

def inspect_sentence(model: tf.keras.Model, token_ids: list[int]) -> dict[str, np.ndarray]:
    if not token_ids:
        raise ValueError("단어 ID가 하나 이상 필요합니다.")

    embedding_layer = model.layers[0]
    lstm_layer = model.layers[1]

    if not isinstance(lstm_layer, tf.keras.layers.LSTM):
        raise ValueError("두 번째 레이어는 LSTM이어야 합니다.")

    if lstm_layer.go_backwards or lstm_layer.stateful:
        raise ValueError("이 관찰 코드는 정방향, stateful=False 모델을 사용합니다.")

    if any(token_id <= 0 or token_id >= embedding_layer.input_dim for token_id in token_ids):
        raise ValueError("패딩 0 없이 어휘 범위 안의 단어 ID를 전달하세요.")

    # 문장 시작 시 h 와 c를 0으로 초기화
    h = tf.zeros((1, lstm_layer.units), dtype=lstm_layer.compute_dtype)
    c = tf.zeros_like(h)

    # 이전 값과 변화량은 출력 시 계산하고 중복 저장하지 않는다.
    history = {name: [] for name in ("f", "i", "g", "o", "h", "c")}

    for token_id in token_ids:
        # 단어 ID 하나 → 단어 벡터 하나
        word_vector = embedding_layer(
            tf.constant([token_id], dtype=tf.int32)
        )
        word_vector = tf.cast(word_vector, lstm_layer.compute_dtype)

        # 고정된 학습 가중치로 계산한다. 추론이므로 드롭아웃은 적용하지 않는다.
        values = calculate_step(lstm_layer.cell, word_vector, h, c)
        h, c = values["h"], values["c"]

        for name, value in values.items():
            history[name].append(value.numpy()[0].copy())

    return {
        name: np.asarray(values)
        for name, values in history.items()
    }
