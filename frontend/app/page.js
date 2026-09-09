"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api } from "../lib/api";

/* ── Upload cloud SVG icon (inline, no dependency) ─────────────── */
function UploadIcon() {
  return (
    <svg
      width="48" height="48" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="16 16 12 12 8 16" />
      <line x1="12" y1="12" x2="12" y2="21" />
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
    </svg>
  );
}

export default function Home() {
  const [books, setBooks]       = useState([]);
  const [uploading, setUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError]       = useState("");
  const fileInputRef = useRef(null);

  const refresh = async () => {
    try   { setBooks(await api.listBooks()); }
    catch (e) { setError(e.message); }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, []);

  const handleFile = async (file) => {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      await api.uploadBook(file);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const onInputChange = (e) => handleFile(e.target.files[0]);

  const onDragOver  = (e) => { e.preventDefault(); setIsDragOver(true); };
  const onDragLeave = ()  => setIsDragOver(false);
  const onDrop      = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <main className="page-main">
      <div className="container">

        {/* ── UPLOAD SECTION ─────────────────────────────────── */}
        <section className="upload-section">
          <div
            className={`dropzone${isDragOver ? " is-over" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <div className="dropzone-icon"><UploadIcon /></div>
            <p className="dropzone-title">Drop a file here, or click to browse</p>
            <p className="dropzone-hint">Supports PDF, DOCX, and TXT files</p>

            <input
              ref={fileInputRef}
              id="file-upload"
              className="upload-file-input"
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={onInputChange}
              disabled={uploading}
            />

            <button
              className="btn-primary"
              type="button"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              {uploading
                ? <><span className="spinner" /> Uploading…</>
                : "Choose file"}
            </button>

            {uploading && (
              <p className="upload-status">Processing in the background…</p>
            )}
            {error && <p className="error-msg">{error}</p>}
          </div>
        </section>

        {/* ── LIBRARY SECTION ────────────────────────────────── */}
        <section className="library-section">
          <h2 className="library-heading">Your Library</h2>

          {books.length === 0 ? (
            <p className="empty-state">No books yet — upload one above to get started.</p>
          ) : (
            <ul className="book-grid">
              {books.map((b) => (
                <li key={b.id} className="book-card">
                  <div className="book-card-top">
                    <span className="book-filename">{b.filename}</span>
                    {statusBadge(b.status)}
                  </div>
                  {b.summary_100w && (
                    <p className="book-summary">{b.summary_100w}</p>
                  )}
                  {b.status === "failed" && (
                    <p className="book-error">{b.error_message}</p>
                  )}
                  {b.status === "ready" && (
                    <Link href={`/book/${b.id}`} className="book-link">
                      Ask questions →
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>

      </div>
    </main>
  );
}

function statusBadge(status) {
  const cls = {
    processing: "badge-processing",
    ready:      "badge-ready",
    failed:     "badge-failed",
  };
  return (
    <span className={`badge ${cls[status] ?? "badge-unknown"}`}>
      {status}
    </span>
  );
}
