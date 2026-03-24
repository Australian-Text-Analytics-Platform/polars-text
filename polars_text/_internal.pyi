from __future__ import annotations

def token_frequencies(texts: list[str]) -> dict[str, int]: ...
def topic_modeling(
    texts: list[str],
    *,
    min_points: int,
    eps: float | None,
    max_terms: int,
    seed: int,
) -> tuple[dict[int, str], list[list[tuple[int, float]]]]: ...
