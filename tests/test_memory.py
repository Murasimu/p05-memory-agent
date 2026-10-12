"""Tests for the memory-enabled conversational agent. Offline, deterministic."""

import json
import os

import pytest

from memory_agent import ConversationAgent, LongTermStore, ShortTermMemory, relevance
from memory_agent.memory import approx_tokens


# ---------------------------------------------------------------- STM
class TestShortTermMemory:
    def test_window_empty(self):
        assert ShortTermMemory().window() == []

    def test_add_and_window(self):
        m = ShortTermMemory()
        m.add("user", "hi")
        m.add("assistant", "hello")
        w = m.window()
        assert [x["role"] for x in w] == ["user", "assistant"]

    def test_bad_role_rejected(self):
        with pytest.raises(ValueError):
            ShortTermMemory().add("robot", "beep")

    def test_compression_kicks_in(self):
        m = ShortTermMemory(max_tokens=60, keep_last=2)
        for i in range(10):
            m.add("user", f"message number {i} " + " ".join(f"w{i}j{k}" for k in range(6)))
        # buffer must be within budget
        assert sum(approx_tokens(x["content"]) for x in m.turns) <= 60
        # recent turns preserved verbatim
        assert any("message number 9" in x["content"] for x in m.turns)
        # old content compressed into digest
        assert m.digest != ""
        w = m.window()
        assert w[0]["role"] == "system" and "compressed" in w[0]["content"]

    def test_keep_last_never_compressed(self):
        m = ShortTermMemory(max_tokens=30, keep_last=3)
        for i in range(8):
            m.add("user", "filler " * 8 + str(i))
        assert len(m.turns) >= 3
        # the loop must never compress below keep_last
        assert len(m.turns) >= 3

    def test_custom_summarizer_used(self):
        calls = []

        def fake_summarizer(old):
            calls.append(len(old))
            return "SUMMARY"

        m = ShortTermMemory(max_tokens=40, keep_last=2, summarizer=fake_summarizer)
        for i in range(6):
            m.add("user", "long message with many words " * 4 + str(i))
        assert calls, "custom summarizer should have been invoked"
        assert "SUMMARY" in m.digest


# ---------------------------------------------------------------- store
class TestLongTermStore:
    def test_save_and_recall(self):
        s = LongTermStore()
        s.save("user's favorite color is teal", "preference")
        s.save("user works as a marine biologist", "fact")
        hits = s.recall("what is my favorite color?")
        assert hits and hits[0][0].text == "user's favorite color is teal"

    def test_recall_scores_sorted(self):
        s = LongTermStore()
        s.save("pizza toppings preference: mushrooms")
        s.save("python programming notes")
        hits = s.recall("mushroom pizza", top_k=2)
        scores = [h[1] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_persistence_and_reload(self, tmp_path):
        p = str(tmp_path / "mem.jsonl")
        s = LongTermStore(path=p)
        s.save("remembered across sessions", "fact")
        s2 = LongTermStore(path=p)
        assert len(s2.records) == 1
        assert s2.records[0].text == "remembered across sessions"

    def test_reload_dedupes_ids(self, tmp_path):
        p = str(tmp_path / "mem.jsonl")
        s = LongTermStore(path=p)
        rec = s.save("only once please", "fact")
        # simulate cross-session double-write of the same record
        with open(p, "a") as fh:
            fh.write(rec.to_json() + "\n")
        s2 = LongTermStore(path=p)
        assert len(s2.records) == 1

    def test_torn_line_skipped(self, tmp_path):
        p = str(tmp_path / "mem.jsonl")
        with open(p, "w") as fh:
            fh.write('{"text": "good", "id": "a"}\n')
            fh.write('{"text": "tor')  # torn write
        s = LongTermStore(path=p)
        assert len(s.records) == 1

    def test_stats(self):
        s = LongTermStore()
        s.save("a", "fact")
        s.save("b", "preference")
        st = s.stats()
        assert st["records"] == 2 and st["kinds"] == {"fact": 1, "preference": 1}


# ---------------------------------------------------------------- agent
class EchoLLM:
    """Returns a canned reply but records the messages it received."""

    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.reply


class TestConversationAgent:
    def test_basic_chat_and_history(self):
        agent = ConversationAgent(EchoLLM("hi there"), LongTermStore())
        r = agent.chat("hello")
        assert r == "hi there"
        assert [t.role for t in agent.history] == ["user", "assistant"]

    def test_memories_injected_into_context(self):
        store = LongTermStore()
        store.save("user is allergic to peanuts", "fact")
        agent = ConversationAgent(EchoLLM("fine"), store)
        agent.chat("I'm allergic to peanuts, what should I eat for lunch?")
        sysmsg = agent.llm.calls[0][0]["content"]
        assert "peanuts" in sysmsg

    def test_irrelevant_memories_not_injected(self):
        store = LongTermStore()
        store.save("user collects vintage stamps about 19th century postal history")
        agent = ConversationAgent(EchoLLM("hm"), store, recall_min_score=0.9)
        agent.chat("what is 2+2?")
        sysmsg = agent.llm.calls[0][0]["content"]
        assert "stamps" not in sysmsg

    def test_preference_extracted_to_long_term(self):
        agent = ConversationAgent(EchoLLM("nice!"), LongTermStore())
        agent.chat("I really love hiking in the mountains on weekends")
        texts = [r.text for r in agent.store.records]
        assert any("love hiking" in t or "hiking in the mountains" in t
                   for t in texts)

    def test_fact_extracted(self):
        agent = ConversationAgent(EchoLLM("ok"), LongTermStore())
        agent.chat("my name is Alex and my job is a pilot")
        kinds = [r.kind for r in agent.store.records]
        assert "fact" in kinds

    def test_dedupe_no_duplicate_memories(self):
        agent = ConversationAgent(EchoLLM("ok"), LongTermStore())
        agent.chat("I really love hiking in the mountains")
        agent.chat("I really love hiking in the mountains")
        agent.chat("I really love hiking in the mountains")
        hiking = [r for r in agent.store.records if "hiking" in r.text]
        assert len(hiking) == 1

    def test_cross_session_recall(self, tmp_path):
        p = str(tmp_path / "session.jsonl")
        a1 = ConversationAgent(EchoLLM("got it"), LongTermStore(path=p))
        a1.chat("my name is Alex")
        # new session, same store file
        a2 = ConversationAgent(EchoLLM("hello"), LongTermStore(path=p))
        a2.chat("hi, do you remember my name?")
        sysmsg = a2.llm.calls[0][0]["content"]
        assert "Alex" in sysmsg

    def test_stats_shape(self):
        agent = ConversationAgent(EchoLLM("x"), LongTermStore())
        agent.chat("hello there friend")
        st = agent.stats()
        assert st["turns_total"] == 2 and "records" in st and "buffer_tokens" in st
