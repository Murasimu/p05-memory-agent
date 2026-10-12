"""Memory-enabled conversational agent.

Short-term buffer + long-term vector recall, context compression,
relevance scoring, cross-session sync (persist to disk, reload later).

Modules:
- memory.py     short-term buffer with token budget + rolling summarization
- store.py      long-term store with vector recall (hash embeddings)
- agent.py      the conversational agent tying them together
"""

from .agent import ConversationAgent, Turn
from .memory import ShortTermMemory
from .store import MemoryRecord, LongTermStore, relevance

__all__ = [
    "ConversationAgent", "Turn",
    "ShortTermMemory",
    "MemoryRecord", "LongTermStore", "relevance",
]
__version__ = "1.0.0"
