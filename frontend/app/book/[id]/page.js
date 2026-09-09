"use client";
import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api } from "../../../lib/api";

/* ── Inline SVG icons — zero dependencies ─────────────────────── */
function ArrowLeftIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

function BookOpenIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

/* ── Avatar initials ── */
function Avatar({ role }) {
  return (
    <div className={`msg-avatar ${role}`}>
      {role === "user" ? "U" : "AI"}
    </div>
  );
}

export default function BookChat() {
  const { id } = useParams();
  const [book, setBook]           = useState(null);
  const [messages, setMessages]   = useState([]);
  const [question, setQuestion]   = useState("");
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    api.getBook(id).then(setBook).catch((e) => setError(e.message));
    api.getChatHistory(id).then(setMessages).catch(() => {});
  }, [id]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const ask = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    const q = question;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    try {
      const res = await api.askQuestion(id, q);
      setMessages((m) => [...m, { role: "assistant", content: res.answer }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page-main">
      <div className="container">

        {/* ── Back nav ── */}
        <Link href="/" className="back-link">
          <ArrowLeftIcon />
          Back to library
        </Link>

        {/* ── Book header + summary ── */}
        {book && (
          <>
            <h1 className="book-title">{book.filename}</h1>

            {book.summary_100w && (
              <div className="summary-card">
                <div className="summary-card-stripe" />
                <div className="summary-card-body">
                  <p className="summary-card-label">
                    <BookOpenIcon />
                    Summary
                  </p>
                  <p className="summary-card-text">{book.summary_100w}</p>
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Chat window ── */}
        <div className="chat-window">
          {messages.length === 0 && !loading && (
            <div className="chat-empty">
              Ask anything about this book — I will answer using only its content.
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role}`}>
              {m.role === "assistant" && <Avatar role="assistant" />}
              <span className={`msg-bubble ${m.role}`}>{m.content}</span>
              {m.role === "user" && <Avatar role="user" />}
            </div>
          ))}

          {loading && (
            <div className="thinking-row">
              <Avatar role="assistant" />
              <div className="thinking-dots">
                <span /><span /><span />
              </div>
              <span className="thinking-label">Thinking…</span>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* ── Input form ── */}
        <form className="chat-form" onSubmit={ask}>
          <input
            className="chat-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask anything about this book…"
            disabled={loading}
            autoComplete="off"
          />
          <button
            className="btn-ask"
            type="submit"
            disabled={loading || !question.trim()}
          >
            <SendIcon />
            Ask
          </button>
        </form>

        {error && (
          <p className="error-msg" style={{ marginTop: "12px" }}>{error}</p>
        )}

      </div>
    </main>
  );
}