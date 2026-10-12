# Memory-Enabled Conversational Agent

**Project 5 / 12 — Agentic AI portfolio.**
*Shows: you can build agents that remember users.*

A conversational agent with two-tier memory: a bounded short-term buffer
with automatic context compression, and a persistent long-term store
with vector recall, relevance scoring, and cross-session sync.

## Why this matters

Stateless chatbots make users repeat themselves forever; naive
"stuff everything in the prompt" agents blow the context window and the
budget. This project implements the memory architecture that production
assistants need:

1. **Short-term buffer with compression** — the rolling window respects a
   token budget; when it overflows, older turns are folded into a running
   digest (pluggable summarizer: LLM in production, extractive by
   default) instead of being silently dropped.
2. **Long-term vector recall** — durable facts/preferences are stored as
   records and retrieved by cosine similarity over embeddings, with a
   relevance threshold so irrelevant memories aren't injected.
3. **Memory extraction** — after each exchange, salient facts and
   preferences are extracted into long-term memory (rule-based extractor
   by default, LLM-extractor slot for production) with near-duplicate
   suppression.
4. **Cross-session sync** — the store persists to JSONL, reloads and
   deduplicates on boot: session 2 remembers session 1.

## Architecture

```
user msg ──> LongTermStore.recall(query) ──> relevant memories (scored)
                     │
     context = system(+memories) + STM.window(digest + recent) + msg
                     │
                   LLM ──> reply
                     │
     STM.add(user, assistant)   (compress if over budget)
     extractor(user, reply) ──> LongTermStore.save (deduped)
                     │
     JSONL persistence ──> next session reloads + dedupes by id
```

- `memory_agent/memory.py` — bounded buffer, token budget, rolling
  digest, pluggable summarizer
- `memory_agent/store.py` — long-term records, hash-embedding vector
  recall, JSONL persistence, dedup, torn-write recovery
- `memory_agent/agent.py` — the turn loop: recall → context → reply →
  extract → persist

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # scripted two-session demo, no network
python demo.py --live     # + a live LLM session using the same memories
```

Demo: session 1 has "Alex" mention their name, job, and food preference;
session 2 boots a *fresh* agent from the same store file and answers
questions that only work if memory survived. The live session proves the
same flow with a real model.

## Tests

```bash
pytest -q
```

20 tests cover: window construction, budget compression (recent turns
kept verbatim, digest created, keep_last floor respected), custom
summarizer integration, save/recall with sorted scores, JSONL
persistence + reload, id dedup, torn-line recovery, stats; agent-level:
history, memory injection into context, irrelevant-memory exclusion,
preference/fact extraction, duplicate suppression, cross-session recall,
stats shape.

## Design decisions

- **Two tiers, not one** — short-term needs to be lossless and ordered;
  long-term needs relevance-based retrieval. Different problems,
  different data structures.
- **Compression is a digest, not deletion** — "conversation so far" is
  preserved in compressed form; the model still knows the topic arc.
- **Recall has a threshold** — injecting every memory every turn is both
  expensive and distracting; scoring keeps the context signal-dense.
- **Dedup on write and on load** — auto-extraction can fire on repeated
  phrasings; near-duplicate suppression (>0.92 similarity) and id-based
  dedup keep the store clean across sessions.
- **Torn writes don't crash the agent** — a truncated JSONL line is
  skipped, not raised.

## Portfolio context

Project 5 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
