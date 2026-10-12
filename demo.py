"""Demo: two sessions with a memory-enabled agent.

Session 1: the user mentions facts and preferences; the agent extracts
them into long-term memory. Session 2 (fresh process-level state, same
store): the user asks something that only works if memory survived.

Usage:
  python demo.py            # scripted LLM, no network
  python demo.py --live     # + live LLM session
"""

from __future__ import annotations

import sys
import tempfile
import os

from memory_agent import ConversationAgent, LongTermStore


def scripted(replies):
    def llm(messages):
        # show what the model "sees" from memory
        sys_msg = messages[0]["content"] if messages else ""
        mem = ""
        for line in sys_msg.splitlines():
            if line.startswith("- ("):
                mem += "    " + line + "\n"
        if mem:
            print(f"    (model sees memories:{chr(10)}{mem}    )")
        return replies.pop(0)
    return llm


def main():
    store_path = os.path.join(tempfile.mkdtemp(), "memory.jsonl")

    print("=" * 62)
    print("SESSION 1 -- user shares facts/preferences")
    print("=" * 62)
    s1 = ConversationAgent(
        scripted([
            "Nice to meet you, Alex! Vegetarian it is.",
            "Tokyo in April -- cherry blossom season, great choice.",
            "A marine biologist who loves vegetarian food -- noted!",
        ]),
        LongTermStore(path=store_path),
    )
    print("user: Hi! My name is Alex, my job is a marine biologist.")
    s1.chat("Hi! My name is Alex, my job is a marine biologist.")
    print("agent: Nice to meet you, Alex! ...")
    print()
    print("user: I really love vegetarian food, by the way.")
    s1.chat("I really love vegetarian food, by the way.")
    print("agent: Vegetarian it is. ...")
    print()
    print("user: I'm planning a trip to Tokyo in April.")
    s1.chat("I'm planning a trip to Tokyo in April.")
    print("agent: Cherry blossom season, great choice. ...")
    print()
    print(f"long-term store: {[r.text for r in LongTermStore(path=store_path).records]}")

    print()
    print("=" * 62)
    print("SESSION 2 -- fresh agent, same store (cross-session sync)")
    print("=" * 62)
    s2 = ConversationAgent(
        scripted([
            "You're Alex, the vegetarian marine biologist -- Tokyo in April sounds perfect!",
            "Given you're vegetarian: try shojin ryori temples' cuisine or veggie ramen!",
        ]),
        LongTermStore(path=store_path),
    )
    print("user: Good morning! Do you remember who I am and what I like?")
    reply = s2.chat("Good morning! Do you remember who I am and what I like?")
    print(f"agent: {reply}")
    print()
    print("user: Any lunch ideas for my Tokyo trip?")
    reply = s2.chat("Any lunch ideas for my Tokyo trip?")
    print(f"agent: (recalled memories drive the suggestion)")
    print()
    print("store stats:", LongTermStore(path=store_path).stats())

    if "--live" in sys.argv:
        print()
        print("=" * 62)
        print("SESSION 3 -- live LLM")
        print("=" * 62)
        from memory_agent.llm import make_llm
        try:
            live = make_llm()
        except RuntimeError as e:
            print(f"skipping live demo: {e}")
            return
        s3 = ConversationAgent(live, LongTermStore(path=store_path))
        for q in [
            "Who am I and what do I do for a living?",
            "What food should I look for in Tokyo?",
        ]:
            print(f"user: {q}")
            print(f"agent: {s3.chat(q)}")
            print(f"  [memory stats] {s3.stats()}")
            print()


if __name__ == "__main__":
    main()
