"""The conversational agent: short-term window + long-term recall +
memory extraction.

Each turn: recall relevant long-term memories for the user's message,
build context (system + memories + compressed window), call the LLM, then
extract salient facts/preferences from the exchange into long-term
memory. Cross-session sync via the JSONL store.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .memory import ShortTermMemory
from .store import LongTermStore, MemoryRecord, relevance

LLM = Callable[[list[dict[str, str]]], str]

SYSTEM_PROMPT = """You are a helpful personal assistant with long-term memory.

Relevant memories about this user (may be empty):
{memories}

Use them when they matter; otherwise answer normally. Do not invent
memories. Keep replies concise."""


@dataclass
class Turn:
    role: str
    content: str
    recalled: list[MemoryRecord] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class ConversationAgent:
    def __init__(
        self,
        llm: LLM,
        store: LongTermStore,
        *,
        max_tokens: int = 800,
        recall_k: int = 3,
        recall_min_score: float = 0.05,
        extractor: Callable[[str, str], list[tuple[str, str]]] | None = None,
        system: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.store = store
        self.stm = ShortTermMemory(max_tokens=max_tokens)
        self.recall_k = recall_k
        self.recall_min_score = recall_min_score
        self.extract = extractor or default_extractor
        self.system = system
        self.history: list[Turn] = []

    def chat(self, user_msg: str) -> str:
        # 1) long-term recall
        recalled = [
            rec for rec, score in self.store.recall(
                user_msg, top_k=self.recall_k, min_score=self.recall_min_score)
        ]

        # 2) build context and call the LLM
        mem_block = "\n".join(
            f"- ({r.kind}) {r.text}" for r in recalled
        ) or "(none)"
        messages = [
            {"role": "system", "content": self.system.format(memories=mem_block)},
            *self.stm.window(),
            {"role": "user", "content": user_msg},
        ]
        reply = self.llm(messages)

        # 3) record the exchange in short-term memory
        self.stm.add("user", user_msg)
        self.stm.add("assistant", reply)
        self.history.append(Turn("user", user_msg, recalled))
        self.history.append(Turn("assistant", reply))

        # 4) extract durable facts into long-term memory
        for text, kind in self.extract(user_msg, reply):
            if relevance(text, text) == 0:  # defensive, never happens
                continue
            # dedupe: skip near-identical existing records
            if any(relevance(text, r.text) > 0.92 for r in self.store.records):
                continue
            self.store.save(text, kind=kind, source="auto-extract")

        return reply

    def stats(self) -> dict:
        s = self.stm.stats()
        s.update(self.store.stats())
        s["turns_total"] = len(self.history)
        return s


# ---------------------------------------------------------------- extraction
_PREF_RE = re.compile(
    r"\b(?:i (?:really )?(?:like|love|prefer|enjoy|hate|dislike)\b[^.!?]*[.!?]?)",
    re.IGNORECASE,
)
_FACT_RE = re.compile(
    r"\b(?:my (?:name|job|role|team|company|city|birthday|dog|cat|car|phone)\b[^.!?]*[.!?]?)",
    re.IGNORECASE,
)


def default_extractor(user_msg: str, reply: str) -> list[tuple[str, str]]:
    """Lightweight rule-based extraction of durable facts/preferences.

    A production system would use an LLM for this; the rules keep the
    demo deterministic and offline while showing the pattern.
    """
    out: list[tuple[str, str]] = []
    for m in _PREF_RE.findall(user_msg):
        t = m.strip().rstrip(".!?")
        if 3 < len(t) < 120:
            out.append((t, "preference"))
    for m in _FACT_RE.findall(user_msg):
        t = m.strip().rstrip(".!?")
        if 3 < len(t) < 120:
            out.append((t, "fact"))
    return out[:4]
