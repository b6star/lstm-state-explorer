import argparse
from pathlib import Path

from state_output import format_states, load_states

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/states.npz")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/states.txt")
    )

    args = parser.parse_args()

    if args.output.suffix.lower() != ".txt":
        parser.error("숫자 출력의 저장 경로는 .txt로 끝나야 합니다.")

    return args

def main() -> None:
    args = parse_args()

    # 저장된 단어 목록과 f, i, g, o, h, c 불러오기
    tokens, states = load_states(args.input)

    # 모든 차원의 이전 값, 현재 값, 변화량을 숫자로 출력하고 저장
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for line in format_states(tokens, states):
            print(line)
            file.write(line + "\n")

    print(f"\n텍스트 저장: {args.output}")

if __name__ == "__main__":
    main()
