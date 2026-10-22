"""Remote embeddings: options, key handling, and runs against a local mock server."""

from __future__ import annotations

import json
import math
import random
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import polars as pl
import polars_text.namespace as namespace_module
import pytest
from polars_text._internal import compiled_features

ENDPOINT = "http://127.0.0.1:9/v1"
KEY_ENV = "POLARS_TEXT_TEST_REMOTE_EMBEDDING_KEY"
SECRET = "sk-sentinel-never-in-a-plan-4242"

needs_remote = pytest.mark.skipif(
    "remote-embedding" not in compiled_features(),
    reason="built without the remote-embedding feature",
)


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        namespace_module,
        "compiled_features",
        lambda: {"embedding", "topic-modeling", "remote-embedding"},
    )
    monkeypatch.setattr(
        namespace_module,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs) or pl.lit(0),
    )
    return calls


def test_remote_options_reach_both_plugins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _capture(monkeypatch)
    monkeypatch.setenv(KEY_ENV, SECRET)
    token = tmp_path / "token"
    token.write_text("t")

    cast(Any, pl.col("text")).text.embedding(
        embedding_endpoint=" https://gateway.example/v1 ",
        embedding_api_key_env=KEY_ENV,
        embedding_remote_model=" served/minilm ",
        embedding_user="person-7",
        embedding_batch_size=64,
        embedding_concurrency=8,
        embedding_timeout_s=30,
    )
    cast(Any, pl.col("text")).text.topic_modeling(
        embedding_endpoint="http://gateway.example:4000/v1",
        embedding_api_key_file=token,
    )

    assert {k: v for k, v in calls[0]["kwargs"].items() if k.startswith("embedding_")} == {
        "embedding_endpoint": "https://gateway.example/v1",
        "embedding_api_key_env": KEY_ENV,
        "embedding_remote_model": "served/minilm",
        "embedding_user": "person-7",
        "embedding_batch_size": 64,
        "embedding_concurrency": 8,
        "embedding_timeout_s": 30.0,
    }
    remote = {k: v for k, v in calls[1]["kwargs"].items() if k.startswith("embedding_")}
    assert remote == {
        "embedding_endpoint": "http://gateway.example:4000/v1",
        "embedding_api_key_file": str(token),
    }
    assert all(SECRET not in repr(call) for call in calls)


def test_without_an_endpoint_the_kwargs_are_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture(monkeypatch)
    cast(Any, pl.col("text")).text.embedding()
    cast(Any, pl.col("text")).text.topic_modeling()
    for call in calls:
        assert not any(key.startswith("embedding_") for key in call["kwargs"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"embedding_user": "x"}, "embedding_user needs embedding_endpoint"),
        ({"embedding_endpoint": "ftp://hunter2.example/v1"}, "http or https"),
        ({"embedding_endpoint": "gateway/v1"}, "http or https"),
        (
            {"embedding_endpoint": "https://me:hunter2@gateway.example/v1"},
            "must not contain credentials",
        ),
        (
            {"embedding_endpoint": "https://gateway.example/v1?key=hunter2"},
            "query or fragment",
        ),
        (
            {"embedding_endpoint": ENDPOINT, "embedding_api_key_env": "SURELY_UNSET_hunter2"},
            "embedding_api_key_env is not set",
        ),
        (
            {"embedding_endpoint": ENDPOINT, "embedding_api_key_file": "/no/hunter2"},
            "embedding_api_key_file does not exist",
        ),
        (
            {
                "embedding_endpoint": ENDPOINT,
                "embedding_api_key_env": KEY_ENV,
                "embedding_api_key_file": "/x",
            },
            "only one of",
        ),
        ({"embedding_endpoint": ENDPOINT, "embedding_batch_size": 0}, "embedding_batch_size"),
        ({"embedding_endpoint": ENDPOINT, "embedding_concurrency": 0}, "embedding_concurrency"),
        ({"embedding_endpoint": ENDPOINT, "embedding_timeout_s": 0}, "embedding_timeout_s"),
    ],
)
def test_invalid_options_name_the_setting_never_its_value(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], message: str
) -> None:
    _capture(monkeypatch)
    monkeypatch.setenv(KEY_ENV, SECRET)
    for method in ("embedding", "topic_modeling"):
        with pytest.raises(ValueError, match=message) as error:
            getattr(cast(Any, pl.col("text")).text, method)(**options)
        assert "hunter2" not in str(error.value)


@needs_remote
def test_the_key_never_appears_in_the_expression_or_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    expr = cast(Any, pl.col("text")).text.topic_modeling(
        embedding_endpoint=ENDPOINT, embedding_api_key_env=KEY_ENV
    )
    lazy = pl.LazyFrame({"text": ["a"]}).select(expr)
    for rendered in (repr(expr), str(expr), lazy.explain()):
        assert SECRET not in rendered
    assert SECRET.encode() not in lazy.serialize()


class _Embeddings(BaseHTTPRequestHandler):
    """An OpenAI-compatible /v1/embeddings answering with deterministic vectors."""

    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(
            {"authorization": self.headers.get("Authorization"), "body": body}
        )
        data = [
            {"object": "embedding", "index": index, "embedding": _vector(text)}
            for index, text in enumerate(body["input"])
        ]
        data.reverse()
        payload = json.dumps({"object": "list", "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_: object) -> None:
        pass


def _vector(text: str) -> list[float]:
    """Unnormalised 384-dimensional vectors: one direction per first word,
    plus small noise seeded by the whole text, so texts group by first word."""
    words = text.split()
    group = sum(map(ord, words[0])) % 7 if words else 0
    noise = random.Random(text)
    return [
        (3.0 if index % 7 == group else 0.0) + noise.uniform(-0.2, 0.2)
        for index in range(384)
    ]


@pytest.fixture
def embeddings_server() -> Iterator[str]:
    _Embeddings.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Embeddings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@needs_remote
def test_embedding_uses_the_remote_server(
    monkeypatch: pytest.MonkeyPatch, embeddings_server: str
) -> None:
    monkeypatch.setenv(KEY_ENV, SECRET)
    frame = pl.DataFrame({"text": ["alpha one", "beta two", None]}).select(
        cast(Any, pl.col("text")).text.embedding(
            embedding_endpoint=embeddings_server,
            embedding_api_key_env=KEY_ENV,
            embedding_user="person-7",
        )
    )
    vectors = frame["text"].to_list()
    assert vectors[2] == []
    for text, vector in zip(["alpha one", "beta two"], vectors[:2], strict=True):
        expected = _vector(text)
        norm = math.sqrt(sum(value * value for value in expected))
        assert vector == pytest.approx([value / norm for value in expected], abs=1e-6)
    request = _Embeddings.seen[0]
    assert request["authorization"] == f"Bearer {SECRET}"
    assert request["body"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert request["body"]["user"] == "person-7"
    assert request["body"]["truncate_prompt_tokens"] == 256


@needs_remote
def test_topic_modeling_embeds_remotely_and_reports_progress(
    embeddings_server: str, tmp_path: Path
) -> None:
    words = ["apple", "river", "stone", "cloud", "piano", "tiger", "maple"]
    documents = [f"{words[index % 7]} story number {index}." for index in range(140)]
    progress = tmp_path / "progress.json"
    result = pl.DataFrame({"text": documents}).select(
        cast(Any, pl.col("text")).text.topic_modeling(
            embedding_endpoint=embeddings_server,
            embedding_batch_size=16,
            embedding_concurrency=3,
            embedding_cache=tmp_path / "embeddings.duckdb",
            min_topic_size=5,
            progress_path=progress,
        )
    )["text"][0]
    assert result["n_segments"] == 140
    assert len(result["topics"]) >= 2
    assert sum(len(request["body"]["input"]) for request in _Embeddings.seen) == 140
    assert max(len(request["body"]["input"]) for request in _Embeddings.seen) <= 16
    assert json.loads(progress.read_text())["step"] == 5
