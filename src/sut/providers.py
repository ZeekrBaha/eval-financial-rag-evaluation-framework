"""
providers.py — Provider abstraction for embedding and generation.

ONE switch between live (OpenAI) and offline (deterministic fixtures).
This is the determinism foundation for the whole project.

Usage:
    from src.sut.providers import get_provider

    provider = get_provider()           # reads EVAL_MODE env var; default "offline"
    provider = get_provider("offline")  # explicit
    provider = get_provider("live")     # requires OPENAI_API_KEY
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Protocol

# Default path for fixtures; can be overridden in OfflineProvider constructor.
_DEFAULT_FIXTURES_PATH = (
    Path(__file__).parent.parent.parent / "datasets" / "fixtures" / "llm.json"
)

# Fixed embedding dimension — must be consistent across the whole project.
_EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Protocol (structural interface) — downstream code depends only on this
# ---------------------------------------------------------------------------


class Provider(Protocol):
    """Structural interface for all provider implementations."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text. Vectors are unit-normalised."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return a text completion for the given prompt."""
        ...


# ---------------------------------------------------------------------------
# Fixture key helper — shared between OfflineProvider and fixture authoring
# ---------------------------------------------------------------------------


def fixture_key(*, system: str | None, prompt: str) -> str:
    """Return a stable, human-readable 16-char hex key for a (system, prompt) pair.

    Used as the key in datasets/fixtures/llm.json.
    Call this to author new fixture entries without running live.
    """
    canonical = json.dumps({"system": system, "prompt": prompt}, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# OfflineProvider — no network, no secrets, fully deterministic
# ---------------------------------------------------------------------------


class OfflineProvider:
    """Deterministic, network-free provider backed by fixture files.

    Embed: seeds a RNG from the text hash → fixed 384-dim unit vector.
    Generate: looks up a recorded output in datasets/fixtures/llm.json.
              Falls back to a deterministic synthetic string if key is absent.
    """

    def __init__(self, fixtures_path: str | None = None) -> None:
        self._fixtures_path = Path(fixtures_path) if fixtures_path else _DEFAULT_FIXTURES_PATH
        self._fixtures: dict[str, str] | None = None  # lazy load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fixtures(self) -> dict[str, str]:
        if self._fixtures is not None:
            return self._fixtures
        if self._fixtures_path.exists():
            raw = self._fixtures_path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"corrupt fixtures file {self._fixtures_path}: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"fixtures file {self._fixtures_path} must contain a JSON object"
                )
            self._fixtures = parsed
        else:
            self._fixtures = {}
        return self._fixtures

    @staticmethod
    def _text_hash_int(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest(), 16)

    # ------------------------------------------------------------------
    # embed
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic unit-norm 384-dim vectors. No network."""
        results: list[list[float]] = []
        for text in texts:
            seed = self._text_hash_int(text)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(_EMBED_DIM)]
            # Normalize to unit length; guard against the (astronomically unlikely) zero vector
            norm = math.sqrt(sum(x * x for x in vec))
            norm = norm or 1.0
            vec = [x / norm for x in vec]
            results.append(vec)
        return results

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return fixture response if available, else a deterministic synthetic string."""
        key = fixture_key(system=system, prompt=prompt)
        fixtures = self._load_fixtures()
        if key in fixtures:
            return fixtures[key]
        # Deterministic fallback — stable across runs, signals clearly it's synthetic
        hash8 = key[:8]
        return f"[offline:{hash8}] no fixture"


# ---------------------------------------------------------------------------
# LiveProvider — OpenAI, lazy import so offline never needs the package/key
# ---------------------------------------------------------------------------


class LiveProvider:
    """OpenAI-backed provider for live runs.

    openai is imported lazily inside each method, so offline mode never
    requires the package or an API key.
    Raises a clear error at call time if OPENAI_API_KEY is missing.
    """

    _EMBED_MODEL = "text-embedding-3-small"
    _CHAT_MODEL = "gpt-4o-mini"

    def _client(self) -> "object":  # returns openai.OpenAI
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: uv add openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Export it before using live mode: export OPENAI_API_KEY=sk-..."
            )
        return openai.OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using text-embedding-3-small."""
        client = self._client()
        response = client.embeddings.create(  # type: ignore[attr-defined]
            model=self._EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion using gpt-4o-mini."""
        client = self._client()
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(mode: str | None = None) -> OfflineProvider | LiveProvider:
    """Return a Provider instance.

    Args:
        mode: "offline" | "live" | None.
              When None, reads EVAL_MODE env var (default "offline").

    Raises:
        ValueError: if mode is not "offline" or "live".
    """
    resolved = mode if mode is not None else os.environ.get("EVAL_MODE", "offline")
    if resolved == "offline":
        return OfflineProvider()
    if resolved == "live":
        return LiveProvider()
    raise ValueError(
        f"unknown provider mode {resolved!r}; expected 'offline' or 'live'"
    )
