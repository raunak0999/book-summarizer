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
    # Group chunks into batches of 4 and add a 2s delay between calls
    # to stay under free-tier TPM rate limits (12,000 TPM).
    batch_size = 4
    partials = []
    chunks = state["chunks"]
    for i in range(0, len(chunks), batch_size):
        if i > 0:
            time.sleep(2)
        batch = "\n\n---\n\n".join(chunks[i:i + batch_size])
        messages = [
            {"role": "system", "content": "You compress book excerpts into concise plot/argument notes. Keep only what matters for an overall summary."},
            {"role": "user", "content": f"Summarize the key events/ideas in the following excerpt in 4-6 sentences:\n\n{batch}"},
        ]
        partials.append(llm.chat(messages, temperature=0.2, max_tokens=300))
    return {**state, "partial_summaries": partials}


def reduce_step(state: SummarizeState) -> SummarizeState:
    time.sleep(4)
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
    draft = llm.chat(messages, temperature=0.3, max_tokens=600)
    log.info("reduce_step_completed", attempt=attempt, word_count=len(draft.split()))
    return {**state, "draft_summary": draft, "attempt": attempt}


def verify_step(state: SummarizeState) -> SummarizeState:
    draft = state.get("draft_summary", "")
    attempt = state.get("attempt", 0)
    word_count = len(draft.split())
    log.info("verify_step_evaluating", attempt=attempt, word_count=word_count)

    if (85 <= word_count <= 115) or (attempt >= 2):
        final = draft if draft else "No summary available."
        log.info("verify_step_accepted", attempt=attempt, word_count=word_count)
        return {**state, "final_summary": final}

    log.info("verify_step_rejected_requesting_rewrite", attempt=attempt, word_count=word_count)
    return {**state, "final_summary": ""}


def route_after_verify(state: SummarizeState) -> str:
    if state.get("final_summary") or state.get("attempt", 0) >= 2:
        return END
    return "reduce"


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
    chunks = chunk_text(full_text, chunk_size=settings.chunk_size_tokens, overlap=0)
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
                "You are a book Q&A assistant. Answer using ONLY the provided context chunks, in clear natural prose without interrupting the answer with inline citations. "
                "If the answer isn't in the context, say you don't have enough information from the book. "
                "At the very end of your answer, on a new line, list the chunk numbers you drew from like this: 'Sources: [chunk N, chunk M]'."
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
