"""Short-term memory: a bounded message buffer with token-budget
compression.

When the buffer exceeds its token budget, the oldest turns are
summarized into a single rolling "conversation so far" digest. The
buffer therefore never overflows, but early context is preserved in
compressed form instead of being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Summarizer = Callable[[list[dict]], str]

APPROX_TOKENS_PER_WORD = 1.3  # rough, English


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * APPROX_TOKENS_PER_WORD) + 1


@dataclass
class ShortTermMemory:
    """Rolling window of turns with a compression digest."""

    max_tokens: int = 800
    keep_last: int = 4  # never summarize the most recent N turns
    _turns: list[dict] = field(default_factory=list)
    _digest: str = ""
    summarizer: Summarizer | None = None  # defaults to naive extractive

    def add(self, role: str, content: str) -> None:
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"bad role {role!r}")
        self._turns.append({"role": role, "content": content})
        self._maybe_compress()

    # ------------------------------------------------------------------
    def _maybe_compress(self) -> None:
        """Compress until the buffer fits the budget.

        The most recent `keep_last` messages are never compressed; older
        ones are folded into the digest (LLM summarizer if provided,
        extractive first-sentences otherwise).
        """
        guard = 0
        while self._over_budget() and len(self._turns) > self.keep_last and guard < 100:
            guard += 1
            overflow = len(self._turns) - self.keep_last
            old = self._turns[:overflow]
            self._turns = self._turns[overflow:]
            self._digest = self._compose_digest(old)

    def _compose_digest(self, old: list[dict]) -> str:
        if self.summarizer is not None:
            fresh = self.summarizer(old)
            return f"{self._digest}\n{fresh}".strip() if self._digest else fresh
        # naive extractive: first sentence of each turn, prefixed by role
        parts = []
        for m in old:
            head = m["content"].split(".")[0].strip()
            if head:
                parts.append(f"{m['role']}: {head}")
        new = " | ".join(parts[-6:])
        return f"{self._digest} | {new}".strip(" |") if self._digest else new

    def _over_budget(self) -> bool:
        total = sum(approx_tokens(m["content"]) for m in self._turns)
        return total > self.max_tokens

    # ------------------------------------------------------------------
    def window(self) -> list[dict]:
        """The context to send to the LLM: digest + recent turns."""
        out: list[dict] = []
        if self._digest:
            out.append({"role": "system",
                       "content": f"Conversation so far (compressed): {self._digest}"})
        out.extend({"role": m["role"], "content": m["content"]} for m in self._turns)
        return out

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def turns(self) -> list[dict]:
        return list(self._turns)

    def stats(self) -> dict:
        return {
            "turns_in_buffer": len(self._turns),
            "digest_tokens": approx_tokens(self._digest) if self._digest else 0,
            "buffer_tokens": sum(approx_tokens(m["content"]) for m in self._turns),
        }
