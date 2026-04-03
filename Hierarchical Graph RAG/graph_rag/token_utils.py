from __future__ import annotations

import math

import tiktoken


def count_tokens(text: str, model: str = "gpt-4.1-mini") -> int:
    if not text:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        pass

    for encoding_name in ("o200k_base", "cl100k_base"):
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            continue

    # Final offline fallback for locked-down environments where tokenizer
    # assets cannot be downloaded. This intentionally overestimates slightly.
    return max(1, math.ceil(len(text) / 4))
