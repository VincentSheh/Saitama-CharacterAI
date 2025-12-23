from collections import deque

from rag.rag_store import RAGStore
from openrouter_client import OpenRouterClient
from memory.ltm_summary import LTMSummary
from memory.summary_updater import SummaryUpdater

from tool.web_search import WebSearchTool

def needs_web_search(user_text: str) -> bool:
    t = user_text.lower()
    triggers = [
        "look up", "search", "latest", "today", "current", "news", "release date",
    ]
    return any(x in t for x in triggers)

def load_persona(path: str = "persona.md") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_messages(
    persona: str,
    stm,
    ltm_user_summary: str,
    ltm_chat_summary: str,
    rag_hits,
    web_hits,
    user_text: str,
):
    rag_block = ""
    if rag_hits:
        rag_block = "\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(rag_hits)])

    web_block = ""
    if web_hits:
        web_block = "\n".join(
            [f"[{i+1}] {h['title']}\n{h['content']}\n{h['url']}" for i, h in enumerate(web_hits)]
        )

    stm_block = ""
    if stm:
        stm_block = "\n".join([f"User: {t['user']}\nSaitama: {t['assistant']}" for t in stm])

    system = (
        f"{persona}\n\n"
        "Stay in character.\n"
        "Be concise.\n"
        "After the main reply add one short inner thought in parentheses.\n"
        "Use RAG_CONTEXT only for One Punch Man canon.\n"
        "Use WEB_CONTEXT only if the user asks to look something up or if the question requires current facts.\n"
        "If something is still unclear, say you're not sure.\n"
    )

    user = (
        "LONG_TERM_USER_SUMMARY:\n"
        f"{ltm_user_summary if ltm_user_summary else '(none)'}\n\n"
        "LONG_TERM_CHAT_SUMMARY:\n"
        f"{ltm_chat_summary if ltm_chat_summary else '(none)'}\n\n"
        "RECENT_CONVERSATION:\n"
        f"{stm_block if stm_block else '(none)'}\n\n"
        "RAG_CONTEXT:\n"
        f"{rag_block if rag_block else '(none)'}\n\n"
        "WEB_CONTEXT:\n"
        f"{web_block if web_block else '(none)'}\n\n"
        "USER_MESSAGE:\n"
        f"{user_text}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def main():
    persona = load_persona("persona.md")

    rag_store = RAGStore.load("rag/artifacts/kb_index")

    ltm = LTMSummary("memory/artifacts/ltm_summary.json")
    ltm.load()

    chat_client = OpenRouterClient(model="openai/gpt-4o-mini")
    updater = SummaryUpdater(OpenRouterClient(model="openai/gpt-4o-mini"))

    stm = deque(maxlen=15)
    turn_count = 0
    web_tool = WebSearchTool()

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        turn_count += 1

        rag_hits = rag_store.retrieve(q, k=5)
        rag_hits = [h for h in rag_hits if h["score"] >= 0.30][:4]
                
        web_hits = []
        if needs_web_search(q):
            web_hits = web_tool.search(q, k=3)
        messages = build_messages(
            persona=persona,
            stm=stm,
            ltm_user_summary=ltm.user_summary,
            ltm_chat_summary=ltm.chat_summary,
            rag_hits=rag_hits,
            web_hits=web_hits,
            user_text=q,
        )

        ans = chat_client.chat(messages, temperature=0.6, max_tokens=350)
        print(f"\nSaitama: {ans}")

        stm.append({"user": q, "assistant": ans})

        # update long-term summaries every turn
        ltm.user_summary = updater.update_user_summary(ltm.user_summary, q, ans)
        ltm.chat_summary = updater.update_chat_summary(ltm.chat_summary, q, ans)
        ltm.save()


if __name__ == "__main__":
    main()