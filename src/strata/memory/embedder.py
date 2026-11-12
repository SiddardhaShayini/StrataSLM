"""Lightweight CPU embedders: zero-dep HashingNgramEmbedder + optional ONNX MiniLM."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from pathlib import Path
from typing import Protocol, Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")


class TextEmbedder(Protocol):
    """Protocol for CPU-friendly unit-normalized text embedders."""

    dim: int

    def encode_one(self, text: str) -> list[float]: ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


def pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class HashingNgramEmbedder:
    """Zero-dependency, <0.2 ms, 0-MB-RAM signed feature-hashing vector embedder."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _WORD_RE.findall(text.lower())
        if not tokens:
            return vec

        features: list[str] = list(tokens)
        features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
        for tok in tokens:
            if len(tok) >= 4:
                features.extend(tok[i : i + 3] for i in range(len(tok) - 2))

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            idx = h % self.dim
            sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            inv = 1.0 / norm
            vec = [v * inv for v in vec]
        return vec

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode_one(t) for t in texts]


class OnnxMiniLMEmbedder:
    """In-process quantized ONNX all-MiniLM-L6-v2 embedder (~90 MB RSS, zero Ollama swaps)."""

    dim: int = 384

    def __init__(
        self,
        model_dir: Path | str,
        *,
        num_threads: int = 4,
        max_length: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        model_path = self._resolve_model_file(self.model_dir)
        tok_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"ONNX MiniLM files missing in {self.model_dir}; expected model_quantized.onnx (or model.onnx) and tokenizer.json"
            )

        # Lazy imports keep CLI cold-start < 300 ms when ONNX is unused.
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, min(num_threads, 6))
        sess_opts.inter_op_num_threads = 1
        sess_opts.enable_mem_pattern = True
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()

    @staticmethod
    def _resolve_model_file(model_dir: Path) -> Path:
        for name in ("model_quantized.onnx", "model_qint8_avx2.onnx", "model.onnx"):
            candidate = model_dir / name
            if candidate.exists():
                return candidate
        return model_dir / "model_quantized.onnx"

    def encode_one(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        np = self._np
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(None, feed)
        token_embeddings = outputs[0]  # (B, seq_len, hidden)

        mask_expanded = attention_mask[:, :, None].astype(np.float32)
        summed = (token_embeddings * mask_expanded).sum(axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized = pooled / norms
        self.dim = int(normalized.shape[1])
        return normalized.astype(np.float32).tolist()


MINILM_REPO_ID = "Xenova/all-MiniLM-L6-v2"
MINILM_ONNX_URL = (
    f"https://huggingface.co/{MINILM_REPO_ID}/resolve/main/onnx/model_quantized.onnx"
)
MINILM_TOKENIZER_URL = (
    f"https://huggingface.co/{MINILM_REPO_ID}/resolve/main/tokenizer.json"
)


def fetch_minilm_onnx(
    model_dir: Path | str = Path(".strata/models/all-MiniLM-L6-v2"),
    *,
    force: bool = False,
    timeout_sec: float = 120.0,
    client: object | None = None,
) -> dict[str, object]:
    """Download quantized all-MiniLM-L6-v2 ONNX weights and tokenizer.json lazily."""
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        ("model_quantized.onnx", MINILM_ONNX_URL),
        ("tokenizer.json", MINILM_TOKENIZER_URL),
    )

    def _download_with(http_client: object) -> dict[str, object]:
        downloaded: list[str] = []
        skipped: list[str] = []
        sizes: dict[str, int] = {}
        for filename, url in artifacts:
            dest = target_dir / filename
            if not force and dest.exists() and dest.stat().st_size > 0:
                skipped.append(filename)
                sizes[filename] = dest.stat().st_size
                continue
            part = target_dir / f".{filename}.part"
            try:
                with http_client.stream("GET", url) as resp:  # type: ignore[attr-defined]
                    resp.raise_for_status()
                    with part.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            if chunk:
                                fh.write(chunk)
                part.replace(dest)
            except Exception:
                part.unlink(missing_ok=True)
                raise
            downloaded.append(filename)
            sizes[filename] = dest.stat().st_size
        return {
            "repo_id": MINILM_REPO_ID,
            "model_dir": str(target_dir),
            "model_path": str(target_dir / "model_quantized.onnx"),
            "tokenizer_path": str(target_dir / "tokenizer.json"),
            "downloaded": downloaded,
            "skipped": skipped,
            "bytes": sizes,
        }

    if client is not None:
        return _download_with(client)

    import httpx  # Lazy import so CLI cold start stays < 300ms

    with httpx.Client(follow_redirects=True, timeout=timeout_sec) as http_client:
        return _download_with(http_client)


def create_embedder(
    *,
    prefer_onnx: bool = True,
    model_dir: Path | str = Path(".strata/models/all-MiniLM-L6-v2"),
    num_threads: int = 4,
) -> TextEmbedder:
    """Select OnnxMiniLMEmbedder if model artifacts & runtime exist, else HashingNgramEmbedder."""
    if prefer_onnx:
        path = Path(model_dir)
        if path.exists() and (path / "tokenizer.json").exists():
            try:
                return OnnxMiniLMEmbedder(path, num_threads=num_threads)
            except (ImportError, FileNotFoundError, OSError):
                pass
    return HashingNgramEmbedder(dim=128)
