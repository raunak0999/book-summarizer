let base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
if (base) {
  base = base.replace(/_/g, "-");
  if (!base.startsWith("http://") && !base.startsWith("https://")) {
    base = `https://${base}`;
  }
  if (!base.includes("localhost") && !base.includes("127.0.0.1") && !base.includes(".")) {
    base = `${base}.onrender.com`;
  }
}
const API_BASE = base;
const TOKEN_KEY = "book_agent_token"; // sessionStorage -> unique per browser tab/window

async function ensureToken() {
  let token = typeof window !== "undefined" ? sessionStorage.getItem(TOKEN_KEY) : null;
  if (!token) {
    const res = await fetch(`${API_BASE}/auth/anonymous`, { method: "POST" });
    const data = await res.json();
    token = data.access_token;
    sessionStorage.setItem(TOKEN_KEY, token);
  }
  return token;
}

async function request(path, options = {}) {
  const token = await ensureToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      Authorization: `Bearer ${token}`,
    },
  });
  if (res.status === 401) {
    sessionStorage.removeItem(TOKEN_KEY);
    throw new Error("Session expired, please retry.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : `Request failed (${res.status})`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  listBooks: () => request("/books"),
  getBook: (id) => request(`/books/${id}`),
  deleteBook: (id) => request(`/books/${id}`, { method: "DELETE" }),
  uploadBook: async (file) => {
    const token = await ensureToken();
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/books`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ? JSON.stringify(body.detail) : `Upload failed (${res.status})`);
    }
    return res.json();
  },
  askQuestion: (bookId, question) =>
    request(`/books/${bookId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  getChatHistory: (bookId) => request(`/books/${bookId}/chat`),
};
