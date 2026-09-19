import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from pathlib import Path
import tensorflow as tf


def build_model(vocab_size: int=10000) -> tf.keras.Model:  # vocab_size: 모델에 사용할 단어 ID의 범위

    # 문장의 길이는 고정하지 않음
    input_layer = tf.keras.Input(
        shape=(None,), 
        dtype="int32"
    )

    # 단어 ID → 128차원 벡터
    # 0(패딩)은 LSTM이 무시하도록 설정
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        mask_zero=True
    )

    # h와 c의 크기가 각각 128인 LSTM 레이어 생성
    lstm_layer = tf.keras.layers.LSTM(128)

    dense_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([
        input_layer,
        embedding_layer,
        lstm_layer,
        dense_layer
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def save_model(model: tf.keras.Model, model_path: Path) -> None:
    # 저장할 폴더가 없으면 생성
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)


def load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

if __name__ == "__main__":
    model = build_model()
    model.summary()