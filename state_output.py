from pathlib import Path
import numpy as np

STATE_NAMES = ("f", "i", "g", "o", "h", "c")

def validate_states(tokens: list[str], states: dict[str, np.ndarray]) -> None:
    if any(name not in states for name in STATE_NAMES):
        raise ValueError(
            "게이트 값이 없는 이전 결과입니다. inspect_states.py를 다시 실행하세요."
        )

    if not tokens:
        raise ValueError("저장된 토큰이 없습니다.")

    shape = states["h"].shape
    if len(shape) != 2 or shape[0] != len(tokens) or shape[1] == 0:
        raise ValueError("토큰 수와 상태 배열의 크기가 맞지 않습니다.")

    if any(states[name].shape != shape for name in STATE_NAMES):
        raise ValueError("f, i, g, o, h, c의 배열 크기가 같아야 합니다.")

def format_states(tokens: list[str], states: dict[str, np.ndarray]):
    validate_states(tokens, states)

    yield "f: 이전 기억 유지 비율 | i: 새 기억 반영 비율"
    yield "g: 새 기억 후보 | o: h로 내보내는 비율"
    yield "previous/current/delta: 직전 토큰 값 / 현재 토큰 값 / 두 값의 차이"
    yield "첫 토큰의 게이트는 이전 값이 없으므로 previous와 delta를 --로 표시합니다."

    for step, token in enumerate(tokens):
        yield f"\n{'=' * 76}"
        yield f"단계 {step + 1} | 단어: {token}"

        for name in STATE_NAMES:
            current = states[name][step]
            previous = states[name][step - 1] if step > 0 else np.zeros_like(current)

            yield f"\n{name}: {len(current)}차원"
            yield f"{'dim':>6} {'previous':>14} {'current':>14} {'delta':>14}"

            # 모든 차원을 한 줄씩 출력
            for dimension, value in enumerate(current):
                if step == 0 and name in ("f", "i", "g", "o"):
                    before, delta = "--", "--"
                else:
                    before = f"{previous[dimension]:+.8f}"
                    delta = f"{value - previous[dimension]:+.8f}"

                yield f"{name}[{dimension:03d}] {before:>14} {value:+14.8f} {delta:>14}"

def print_states(tokens: list[str], states: dict[str, np.ndarray]) -> None:
    for line in format_states(tokens, states):
        print(line)

def save_states(tokens: list[str], states: dict[str, np.ndarray], output_path: Path) -> None:
    if output_path.suffix != ".npz":
        raise ValueError("저장 경로는 .npz로 끝나야 합니다.")

    validate_states(tokens, states)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 전체 게이트와 상태의 원래 배열 값 저장
    np.savez_compressed(
        output_path,
        tokens=np.asarray(tokens, dtype=str),
        **states
    )

    # 콘솔 출력과 같은 내용을 텍스트로 저장
    with output_path.with_suffix(".txt").open("w", encoding="utf-8") as file:
        for line in format_states(tokens, states):
            file.write(line + "\n")

def load_states(input_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(input_path, allow_pickle=False) as data:
        tokens = data["tokens"].tolist()
        states = {name: data[name] for name in data.files if name != "tokens"}

    validate_states(tokens, states)
    return tokens, states
