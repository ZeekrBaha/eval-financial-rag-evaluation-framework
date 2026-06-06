"""
Tests for src/sut/providers.py — T2: Provider abstraction (live/offline).

TDD order: these tests are written BEFORE the implementation.
All offline tests must pass without network access and without openai installed.
"""

import importlib
import math
import os
import sys

import pytest

from src.sut.providers import (
    OfflineProvider,
    LiveProvider,
    get_provider,
    fixture_key,
)


# ---------------------------------------------------------------------------
# OfflineProvider — embed
# ---------------------------------------------------------------------------


class TestOfflineEmbed:
    def setup_method(self) -> None:
        self.provider = OfflineProvider()

    def test_embed_returns_list_of_lists(self) -> None:
        result = self.provider.embed(["hello world"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_embed_dimension_is_384(self) -> None:
        result = self.provider.embed(["some financial text"])
        assert len(result[0]) == 384

    def test_embed_unit_norm(self) -> None:
        result = self.provider.embed(["normalize me"])
        vec = result[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_embed_determinism(self) -> None:
        """Same text must produce identical vector across two calls."""
        text = "Apple Inc. revenue Q4 2023"
        v1 = self.provider.embed([text])[0]
        v2 = self.provider.embed([text])[0]
        assert v1 == v2

    def test_embed_multiple_texts(self) -> None:
        texts = ["first text", "second text", "third text"]
        result = self.provider.embed(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 384

    def test_embed_different_texts_differ(self) -> None:
        """Different texts should (with overwhelming probability) produce different vectors."""
        v1 = self.provider.embed(["revenue growth"])[0]
        v2 = self.provider.embed(["debt to equity ratio"])[0]
        assert v1 != v2


# ---------------------------------------------------------------------------
# OfflineProvider — generate
# ---------------------------------------------------------------------------


class TestOfflineGenerate:
    def setup_method(self) -> None:
        self.provider = OfflineProvider()

    def test_generate_returns_string(self) -> None:
        result = self.provider.generate("What is the P/E ratio?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_determinism(self) -> None:
        """Same prompt + system must produce identical output across two calls."""
        prompt = "Explain free cash flow"
        system = "You are a financial analyst."
        r1 = self.provider.generate(prompt, system=system)
        r2 = self.provider.generate(prompt, system=system)
        assert r1 == r2

    def test_generate_determinism_no_system(self) -> None:
        """Determinism also holds when system is None."""
        prompt = "What is EBITDA?"
        r1 = self.provider.generate(prompt)
        r2 = self.provider.generate(prompt)
        assert r1 == r2

    def test_generate_missing_key_returns_synthetic(self) -> None:
        """When no fixture matches, return a deterministic synthetic string."""
        # Use a unique prompt unlikely to be in fixtures
        result = self.provider.generate("zzzuniqueprompt_no_fixture_xyzzy_99")
        assert "[offline:" in result
        assert "no fixture" in result

    def test_generate_fixture_hit(self, tmp_path: "pytest.TempPathFactory", monkeypatch: pytest.MonkeyPatch) -> None:
        """Fixture loaded from llm.json is returned when key matches."""
        import json

        # Compute the key for our test input
        system = "sys"
        prompt = "test fixture prompt"
        key = fixture_key(system=system, prompt=prompt)
        expected_output = "This is the recorded fixture response."

        # Write a temp fixtures file
        fixtures_file = tmp_path / "llm.json"
        fixtures_file.write_text(json.dumps({key: expected_output}))

        # Point OfflineProvider at our temp fixtures file
        provider = OfflineProvider(fixtures_path=str(fixtures_file))
        result = provider.generate(prompt, system=system)
        assert result == expected_output


# ---------------------------------------------------------------------------
# No-network assertion for offline mode
# ---------------------------------------------------------------------------


class TestOfflineNoNetwork:
    """Prove OfflineProvider never touches openai, even if it were importable."""

    def test_offline_embed_does_not_import_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch sys.modules so any openai import would raise ImportError."""
        monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
        provider = OfflineProvider()
        # Must not raise despite openai being "unavailable"
        result = provider.embed(["test text"])
        assert len(result[0]) == 384

    def test_offline_generate_does_not_import_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
        provider = OfflineProvider()
        result = provider.generate("some prompt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------


class TestGetProvider:
    def test_default_is_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no env var, get_provider() must return OfflineProvider."""
        monkeypatch.delenv("EVAL_MODE", raising=False)
        provider = get_provider()
        assert isinstance(provider, OfflineProvider)

    def test_offline_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVAL_MODE", "offline")
        provider = get_provider()
        assert isinstance(provider, OfflineProvider)

    def test_mode_arg_offline(self) -> None:
        provider = get_provider(mode="offline")
        assert isinstance(provider, OfflineProvider)

    def test_mode_arg_live_returns_live_type(self) -> None:
        """get_provider('live') returns LiveProvider — do NOT call its methods (no key in CI)."""
        provider = get_provider(mode="live")
        assert isinstance(provider, LiveProvider)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            get_provider(mode="bogus")


# ---------------------------------------------------------------------------
# fixture_key helper
# ---------------------------------------------------------------------------


class TestFixtureKey:
    def test_key_is_string(self) -> None:
        key = fixture_key(system=None, prompt="hello")
        assert isinstance(key, str)

    def test_key_determinism(self) -> None:
        k1 = fixture_key(system="sys", prompt="prompt")
        k2 = fixture_key(system="sys", prompt="prompt")
        assert k1 == k2

    def test_key_differs_by_system(self) -> None:
        k1 = fixture_key(system="a", prompt="same")
        k2 = fixture_key(system="b", prompt="same")
        assert k1 != k2

    def test_key_differs_by_prompt(self) -> None:
        k1 = fixture_key(system=None, prompt="prompt_a")
        k2 = fixture_key(system=None, prompt="prompt_b")
        assert k1 != k2
