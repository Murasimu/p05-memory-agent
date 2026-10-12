"""Long-term memory: persistent records with vector recall.

Records are (text, metadata) with timestamps. Recall is by cosine
similarity over hash embeddings (same trick as P2: deterministic,
offline, zero-dependency; swap in a real embedder for production).
Cross-session sync = JSONL persistence: append on save, reload on boot,
records deduplicated by id.
"""

from __future__ import annotations

import json
import math
import os
import time
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

DIM = 256


def _embed_one(text: str) -> list[float]:
    vec = [0.0] * DIM
    counts: dict[int, int] = {}
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        counts[h % DIM] = counts.get(h % DIM, 0) + 1
    for b, c in counts.items():
        vec[b] = 1.0 + math.log(c)
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class MemoryRecord:
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str = "fact"  # fact | preference | event
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class LongTermStore:
    """Vector-indexed persistent memory (JSONL file per session/user)."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.records: list[MemoryRecord] = []
        self._vecs: list[list[float]] = []
        if path and os.path.exists(path):
            self._load()

    # ------------------------------------------------------------ CRUD
    def save(self, text: str, kind: str = "fact", **meta: Any) -> MemoryRecord:
        rec = MemoryRecord(text=text, kind=kind, meta=meta)
        self._append(rec)
        return rec

    def _append(self, rec: MemoryRecord) -> None:
        self.records.append(rec)
        self._vecs.append(_embed_one(rec.text))
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")

    def _load(self) -> None:
        seen: set[str] = set()
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn write: skip, don't crash the agent
                if d.get("id") in seen:
                    continue  # dedupe across sessions
                seen.add(d.get("id"))
                rec = MemoryRecord(
                    text=d["text"], id=d["id"], kind=d.get("kind", "fact"),
                    ts=d.get("ts", 0.0), meta=d.get("meta", {}),
                )
                self.records.append(rec)
                self._vecs.append(_embed_one(rec.text))

    # ------------------------------------------------------------ recall
    def recall(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[MemoryRecord, float]]:
        """Most relevant records for `query`, with similarity scores."""
        q = _embed_one(query)
        scored = [
            (rec, cosine(q, v))
            for rec, v in zip(self.records, self._vecs)
        ]
        scored = [s for s in scored if s[1] >= min_score]
        scored.sort(key=lambda s: -s[1])
        return scored[:top_k]

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for r in self.records:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        return {"records": len(self.records), "kinds": kinds}


def relevance(query: str, text: str) -> float:
    """Quick relevance check between a query and one memory text."""
    return cosine(_embed_one(query), _embed_one(text))
