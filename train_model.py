import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_io import build_model, save_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/lstm.keras")
    )

    return parser.parse_args()

def load_data(max_len: int, limit: int=0):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=10000
    )

    if limit > 0:
        x_train = x_train[:limit]
        y_train = y_train[:limit]

    # 긴 리뷰는 뒤를 자르고, 짧은 리뷰는 뒤에 0을 채움
    x_train = pad_sequences(
        x_train,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    x_test = pad_sequences(
        x_test,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    return x_train, y_train, x_test, y_test

def train_model(model, x_train, y_train, batch_size=32, epochs=3):
    return model.fit(
        x_train, 
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )

def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(42)

    x_train, y_train, x_test, y_test = load_data(args.max_len)

    model = build_model()

    train_model(
        model,
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    loss, accuracy = model.evaluate(
        x_test, 
        y_test,
        batch_size=args.batch_size
    )

    print(f"테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.2%}")

    save_model(model, args.model)
    print(f"모델 저장 완료: {args.model}")

if __name__ == "__main__":
    main()


