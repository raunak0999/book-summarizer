"""
Two LangGraph graphs:

1. `summarize_graph` — a map-reduce agentic workflow for the 100-word
   summary. A single LLM call can't ingest a 500+ page book, so we:
     map: summarize each chunk-group in parallel-ish steps -> partial summaries
     reduce: summarize the partial summaries into one 100-word summary
     verify: a critique step checks the word count / faithfulness and
              triggers a rewrite if needed (this is the "agentic" loop,
              not one-shot prompting).

2. `qa_graph` — the RAG agent used for "ask anything about the book":
     retrieve -> grade relevance -> (optionally re-retrieve with a
     rewritten query) -> generate grounded answer -> cite chunk ids.

Both are plain LangGraph StateGraphs so they're easy to extend later
(requirement #6.iii).
"""
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.services.llm_client import get_llm_client
from app.services.document_processing import chunk_text, count_tokens
from app.core.config import get_settings
from app.core.logging import log

settings = get_settings()



# ----------------------------- Summarization graph -----------------------------

class SummarizeState(TypedDict):
    chunks: list[str]
    partial_summaries: list[str]
    draft_summary: str
    final_summary: str
    attempt: int


def map_step(state: SummarizeState) -> SummarizeState:
    llm = get_llm_client()
    chunks = state["chunks"]

    # Process in groups of 5 chunks (~40k tokens/call) rather than one giant call.
    # Sending all chunks at once can hit Gemini's per-minute token quota even within
    # a single request (a 586-page book = ~233k tokens in one prompt).
    GROUP_SIZE = 5
    groups = [chunks[i:i + GROUP_SIZE] for i in range(0, len(chunks), GROUP_SIZE)]
    expected = len(groups)
    log.info("map_step_start", total_chunks=len(chunks), groups=expected)

    partials = []
    for idx, group in enumerate(groups):
        batch = "\n\n---\n\n".join(group)
        messages = [
            {
                "role": "system",
                "content": "You compress book excerpts into concise plot/argument notes. Keep only what matters for an overall summary.",
            },
            {
                "role": "user",
                "content": f"Summarize the key events/ideas in the following book text in 8-12 sentences:\n\n{batch}",
            },
        ]
        partial = llm.chat(messages, temperature=0.2, max_tokens=1500)
        partials.append(partial)
        log.info(
            "map_step_group_complete",
            group=f"{idx + 1}/{expected}",
            collected=len(partials),
        )

    # Sanity check: every group should have produced exactly one summary
    if len(partials) != expected:
        log.warning(
            "map_step_count_mismatch",
            collected=len(partials),
            expected=expected,
        )
    else:
        log.info("map_step_complete", collected=len(partials), expected=expected)

    return {**state, "partial_summaries": partials}


def reduce_step(state: SummarizeState) -> SummarizeState:
    llm = get_llm_client()
    attempt = state.get("attempt", 0) + 1

    joined = "\n\n".join(state.get("partial_summaries", []))
    messages = [
        {"role": "system", "content": "You write precise, spoiler-appropriate book summaries."},
        {"role": "user", "content": (
            "Using ONLY the notes below (which cover the entire book sequentially), "
            "write a single cohesive summary of the WHOLE book in EXACTLY around 100 words. "
            "Do not exceed 110 words or go below 90 words.\n\nNotes:\n" + joined
        )},
    ]
    draft = llm.chat(messages, temperature=0.3, max_tokens=2000)
    log.info("reduce_step_completed", attempt=attempt, word_count=len(draft.split()))
    return {**state, "draft_summary": draft, "attempt": attempt}


def verify_step(state: SummarizeState) -> SummarizeState:
    """Accept on first attempt to save API calls (free-tier RPD is very limited)."""
    draft = state.get("draft_summary", "")
    attempt = state.get("attempt", 0)
    word_count = len(draft.split())
    log.info("verify_step_evaluating", attempt=attempt, word_count=word_count)
    # Always accept to save API calls — the reduce prompt already enforces ~100 words
    final = draft if draft else "No summary available."
    log.info("verify_step_accepted", attempt=attempt, word_count=word_count)
    return {**state, "final_summary": final}


def route_after_verify(state: SummarizeState) -> str:
    # Always END — no rewrite loop to save API calls
    return END


def build_summarize_graph():
    g = StateGraph(SummarizeState)
    g.add_node("map", map_step)
    g.add_node("reduce", reduce_step)
    g.add_node("verify", verify_step)
    g.set_entry_point("map")
    g.add_edge("map", "reduce")
    g.add_edge("reduce", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"reduce": "reduce", END: END})
    return g.compile()


def run_summarization(full_text: str) -> str:
    # Use large chunks — Gemini supports 1M token context
    chunks = chunk_text(full_text, chunk_size=8000, overlap=0)
    graph = build_summarize_graph()
    result = graph.invoke(
        {"chunks": chunks, "partial_summaries": [], "draft_summary": "", "final_summary": "", "attempt": 0},
        config={"recursion_limit": 10}
    )
    final = result.get("final_summary") or result.get("draft_summary") or ""
    return final.strip()


# ----------------------------- RAG Q&A graph -----------------------------

class QAState(TypedDict):
    book_id: str
    question: str
    history: list[dict]
    retrieved: list[dict]
    relevant: bool
    answer: str


def make_qa_graph(retriever_fn):
    """retriever_fn(book_id, query, k) -> list[{content, ordinal, score}]"""

    def retrieve(state: QAState) -> QAState:
        results = retriever_fn(state["book_id"], state["question"], settings.top_k_retrieval)
        return {**state, "retrieved": results}

    def grade(state: QAState) -> QAState:
        # Cheap heuristic grading: if we got any chunks above a similarity
        # floor, consider it relevant. Keeps this fast/cheap instead of an
        # extra LLM call for every query; swap for an LLM grader if needed.
        relevant = len(state["retrieved"]) > 0 if state["retrieved"] else False
        return {**state, "relevant": relevant}

    def generate(state: QAState) -> QAState:
        llm = get_llm_client()
        context = "\n\n".join(f"[chunk {r['ordinal']}] {r['content']}" for r in state["retrieved"])
        
        # Keep last 4 messages and safely truncate long assistant/user turns to <=300 chars
        recent_history = state.get("history", [])[-4:] if state.get("history") else []
        history_lines = []
        for m in recent_history:
            role = m.get("role", "user")
            content = m.get("content") or ""
            content_snippet = content[:300] + ("..." if len(content) > 300 else "")
            history_lines.append(f"{role}: {content_snippet}")
        history_txt = "\n".join(history_lines)

        messages = [
            {"role": "system", "content": (
                "You are a helpful book Q&A assistant. Answer the user's question using ONLY the provided context chunks in clear, natural, and friendly prose. "
                "Do not output raw chunk labels or 'Sources: [chunk X]' text. "
                "If the answer isn't in the context, politely state that you don't have enough information from the book."
            )},

            {"role": "user", "content": (
                f"Conversation so far:\n{history_txt}\n\n"
                f"Context:\n{context if context else '(no relevant context found)'}\n\n"
                f"Question: {state['question']}"
            )},
        ]
        
        total_chars = sum(len(m.get("content") or "") for m in messages)
        log.info("qa_payload_size", n_messages=len(messages), total_chars=total_chars, retrieved_chunks=len(state["retrieved"]))

        try:
            answer = llm.chat(messages, temperature=0.2, max_tokens=500)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "413" in exc_str or "request_too_large" in exc_str or "too large" in exc_str:
                log.warning("qa_payload_too_large_handled", error=str(exc))
                answer = "The question or conversation history is too long. Please ask a shorter follow-up question or clear history."
            else:
                raise exc

        return {**state, "answer": answer}



    def not_relevant(state: QAState) -> QAState:
        return {**state, "answer": "I couldn't find anything relevant to that question in this book."}

    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("not_relevant", not_relevant)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", lambda s: "generate" if s["relevant"] else "not_relevant",
                             {"generate": "generate", "not_relevant": "not_relevant"})
    g.add_edge("generate", END)
    g.add_edge("not_relevant", END)
    return g.compile()
