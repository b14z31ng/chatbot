import { useState, useRef, useEffect } from "react";
import { chat } from "../services/api";
import UploadPanel from "./UploadPanel";

export default function ChatBox({ token, onLogout }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [showUpload, setShowUpload] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res = await chat(text, token, conversationId);
      if (res.conversation_id) {
        setConversationId(res.conversation_id);
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer || "Not in knowledge base.",
          citations: res.citations,
          confidence: res.confidence,
        },
      ]);
    } catch (err) {
      if (err.message === "SESSION_EXPIRED") {
        onLogout();
        return;
      }
      setMessages((prev) => [
        ...prev,
        { role: "error", content: err.message },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const confidenceColor = (c) => {
    if (c >= 0.7) return "confidence-high";
    if (c >= 0.4) return "confidence-mid";
    return "confidence-low";
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-left">
          <span className="header-icon">🤖</span>
          <h1>RAG Chatbot</h1>
        </div>
        <div className="header-right">
          <button
            className={`upload-toggle-btn ${showUpload ? "active" : ""}`}
            onClick={() => setShowUpload(!showUpload)}
            title="Upload PDF"
          >
            📄
          </button>
          <button className="logout-btn" onClick={onLogout}>
            Sign Out
          </button>
        </div>
      </header>

      {showUpload && <UploadPanel token={token} onLogout={onLogout} />}

      <div className="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <p>Start a conversation by asking a question.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message message-${msg.role}`}>
            <div className="message-bubble">
              <p className="message-text">{msg.content}</p>
              {msg.role === "assistant" && (
                <div className="message-meta">
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="citations">
                      <span className="meta-label">Sources:</span>
                      {msg.citations.map((c, j) => (
                        <span key={j} className="citation-tag">
                          {c}
                        </span>
                      ))}
                    </div>
                  )}
                  {msg.confidence !== undefined && (
                    <span
                      className={`confidence-badge ${confidenceColor(msg.confidence)}`}
                    >
                      {Math.round(msg.confidence * 100)}% confidence
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message message-assistant">
            <div className="message-bubble">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-area" onSubmit={sendMessage}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question…"
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>
    </div>
  );
}
