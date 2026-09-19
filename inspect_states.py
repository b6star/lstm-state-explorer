import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
from pathlib import Path

from tensorflow.keras.datasets import imdb

from model_io import load_model
from text_processing import encode_sentence
from state_tracker import inspect_sentence
from state_output import print_states, save_states

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/lstm.keras")
    )
    parser.add_argument(
        "--text",
        type=str,
        default="this movie is not good"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/states.npz")
    )

    args = parser.parse_args()

    if not args.text.strip():
        parser.error("문장을 입력하세요.")

    if args.output.suffix != ".npz":
        parser.error("저장 경로는 .npz로 끝나야 합니다.")

    return args

def main() -> None:
    args = parse_args()

    # 학습된 모델과 IMDB 단어 사전 불러오기
    model = load_model(args.model)
    word_index = imdb.get_word_index()

    # 문장을 단어 ID로 변환
    tokens, token_ids = encode_sentence(
        args.text,
        word_index,
        vocab_size=model.layers[0].input_dim
    )

    print(f"문장: {args.text}")
    print(f"단어 ID: {token_ids}")

    for token, token_id in zip(tokens, token_ids):
        if token_id == 2:
            print(f"미등록 단어로 처리됨: {token}")

    # 단어별 f, i, g, o와 h, c 수집
    states = inspect_sentence(model, token_ids)

    # 전체 차원 출력 및 저장
    print_states(tokens, states)
    save_states(tokens, states, args.output)

    print(f"\n배열 저장: {args.output}")
    print(f"텍스트 저장: {args.output.with_suffix('.txt')}")

if __name__ == "__main__":
    main()