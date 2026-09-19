from pathlib import Path
import numpy as np

STATE_NAMES = ("f", "i", "g", "o", "c", "h")

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

    yield "토큰별 현재 값: 각 행은 차원, 각 열은 토큰입니다."

    for name in STATE_NAMES:
        yield from format_single_state(tokens, states, name)

def format_single_state(tokens: list[str], states: dict[str, np.ndarray], name: str):
    validate_states(tokens, states)

    if name not in STATE_NAMES:
        raise ValueError(f"알 수 없는 상태 이름입니다: {name}")

    values = states[name]
    column_width = 14

    yield f"\n{'=' * 76}"
    yield f"{name}: {values.shape[1]}차원"
    yield "현재 값"
    yield f"{'dim':>8}" + "".join(f"{token:>{column_width}}" for token in tokens)

    for dimension, row in enumerate(values.T):
        values_text = "".join(f"{value:+{column_width}.8f}" for value in row)
        yield f"{name}[{dimension:03d}]" + values_text

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

    save_separate_state_files(tokens, states, output_path.with_suffix(""))

def save_separate_state_files(
    tokens: list[str],
    states: dict[str, np.ndarray],
    output_stem: Path
) -> None:
    validate_states(tokens, states)

    for name in STATE_NAMES:
        output_path = output_stem.parent / f"{output_stem.name}_{name}.txt"
        with output_path.open("w", encoding="utf-8") as file:
            for line in format_single_state(tokens, states, name):
                file.write(line + "\n")

def load_states(input_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(input_path, allow_pickle=False) as data:
        tokens = data["tokens"].tolist()
        states = {name: data[name] for name in data.files if name != "tokens"}

    validate_states(tokens, states)
    return tokens, states
