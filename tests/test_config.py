"""Tests for StrataConfig and OLLAMA_HOST normalization."""

from pathlib import Path

from strata.config import StrataConfig, normalize_ollama_host


def test_normalize_ollama_host() -> None:
    assert normalize_ollama_host(None) == "http://127.0.0.1:11434"
    assert normalize_ollama_host("") == "http://127.0.0.1:11434"
    assert normalize_ollama_host("127.0.0.1:11434/") == "http://127.0.0.1:11434"
    assert normalize_ollama_host("http://0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert normalize_ollama_host("https://localhost:11434/") == "https://localhost:11434"


def test_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11435")
    monkeypatch.setenv("STRATA_NUM_CTX", "2048")
    monkeypatch.setenv("STRATA_NUM_THREAD", "6")
    monkeypatch.setenv("STRATA_PRIMARY_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("STRATA_DATA_DIR", "custom_data")

    cfg = StrataConfig.from_env()
    assert cfg.ollama_host == "http://127.0.0.1:11435"
    assert cfg.num_ctx == 2048
    assert cfg.num_thread == 6
    assert cfg.primary_model == "qwen3:0.6b"
    assert cfg.db_path == Path("custom_data/strata.db")
