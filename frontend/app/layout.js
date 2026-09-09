import "./globals.css";

export const metadata = {
  title: "Book Summarizer & Query Agent",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="app-shell">
        <header className="app-header">
          <span className="app-header-icon">📚</span>
          <span className="app-header-title">Book Summarizer &amp; Query Agent</span>
        </header>
        {/* No wrapper here — each page owns its own container */}
        {children}
      </body>
    </html>
  );
}
