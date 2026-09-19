def encode_sentence(
    sentence: str,
    word_index: dict[str, int],
    vocab_size: int=10000
) -> tuple[list[str], list[int]]:
    
    words = sentence.lower().split()

    if not words:
        raise ValueError("문장을 .")

    # IMDB의 문장 시작 토큰
    tokens = ["<START>"]
    token_ids = [1]

    for word in words:
        index = word_index.get(word)

        if index is None:
            token_id = 2            # 사전에 없는 단어
        else:
           token_id = index + 3     # IMDB 기본 인덱스 규칙

           if token_id >= vocab_size:
               token_id = 2         # 사용할 어휘 범위 밖의 단어

        tokens.append(word)
        token_ids.append(token_id)

    return tokens, token_ids
    

if __name__ == "__main__":
    word_index = { "this": 11, "movie": 17, "good": 49 }

    tokens, token_ids = encode_sentence(
        "this movie good unknown",
        word_index
    )

    print(tokens)
    print(token_ids)
