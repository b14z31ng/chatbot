import { useState, useRef, useEffect, useCallback } from "react";
import { getChatMessages, sendChatMessage } from "../services/api";
import MessageBubble from "./MessageBubble";
import UploadArea from "./UploadArea";
import { toast } from "./Toast";

function TypingIndicator() {
  return (
    <div className="message message-assistant">
      <div className="msg-avatar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2a7 7 0 00-7 7c0 2.4 1.2 4.5 3 5.7V17a2 2 0 002 2h4a2 2 0 002-2v-2.3c1.8-1.2 3-3.3 3-5.7a7 7 0 00-7-7z" />
        </svg>
      </div>
      <div className="message-bubble">
        <div className="typing-indicator">
          <span /><span /><span />
        </div>
      </div>
    </div>
  );
}

export default function ChatWindow({ chatId, chatTitle, token, username, onLogout, onChatUpdated }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [activeDocuments, setActiveDocuments] = useState([]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const textareaRef = useRef(null);
  const submittingRef = useRef(false);
  const lastSubmitRef = useRef(0);

  useEffect(() => {
    console.log("ChatBox mounted");
    return () => console.log("ChatBox unmounted");
  }, []);

  useEffect(() => {
    console.log("showUpload:", showUpload);
  }, [showUpload]);

  useEffect(() => {
    console.log("chatId changed:", chatId);
  }, [chatId]);

  // ── Scroll to bottom ─────────────────────────────────────────────────────
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, scrollToBottom]);

  // ── Chat switch: reset all local state ───────────────────────────────────
  useEffect(() => {
    setMessages([]);
    setActiveDocuments([]);
    setShowUpload(false);   // only automatic close: when switching chats
    setInput("");
    if (!chatId) return;

    setLoadingHistory(true);
    getChatMessages(chatId, token)
      .then((msgs) => setMessages(msgs))
      .catch((err) => {
        if (err.message === "SESSION_EXPIRED") onLogout();
        else toast("Failed to load messages", "error");
      })
      .finally(() => setLoadingHistory(false));
  }, [chatId, token, onLogout]);

  // ── Escape closes upload panel ────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") setShowUpload(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // ── Textarea auto-resize ──────────────────────────────────────────────────
  const handleInputChange = (e) => {
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    setInput(e.target.value);
  };

  // ── Send message ──────────────────────────────────────────────────────────
  const sendMessage = async (e) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || loading || !chatId) return;

    const now = Date.now();
    if (submittingRef.current || now - lastSubmitRef.current < 300) return;
    submittingRef.current = true;
    lastSubmitRef.current = now;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const assistantMsg = await sendChatMessage(chatId, text, token);
      setMessages((prev) => [...prev, { ...assistantMsg, citations: [] }]);
      onChatUpdated?.();
    } catch (err) {
      if (err.message === "SESSION_EXPIRED") { onLogout(); return; }
      setMessages((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, role: "error", content: err.message },
      ]);
      toast(err.message, "error");
    } finally {
      setLoading(false);
      submittingRef.current = false;
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ── Document change / upload success callbacks ────────────────────────────
  const handleDocumentsChange = useCallback((docs) => {
    setActiveDocuments(docs);
    onChatUpdated?.();
  }, [onChatUpdated]);

  const handleUploadSuccess = useCallback(() => {
    // Close panel 800ms after a successful upload
    setTimeout(() => setShowUpload(false), 800);
  }, []);

  // ── No chat selected ──────────────────────────────────────────────────────
  if (!chatId) {
    return (
      <div className="chat-window chat-window-empty">
        <div className="welcome-state">
          <div className="welcome-icon">✦</div>
          <h2>Welcome to RAG Chat</h2>
          <p>Select a conversation from the sidebar or create a new chat to get started.</p>
          <p className="welcome-hint">Upload PDFs to your chat, then ask questions — answers are grounded in your documents.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-window">

      {/* Top Bar */}
      <header className="chat-topbar">
        <div className="topbar-left">
          <h2 className="topbar-title" title={chatTitle}>{chatTitle || "New Chat"}</h2>
          {activeDocuments.length > 0 && (
            <div className="topbar-docs">
              {activeDocuments.map((doc) => (
                <span key={doc.id} className="topbar-doc-pill" title={doc.filename}>
                  📎 {doc.filename}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="topbar-right">
          <button
            className={`topbar-btn ${showUpload ? "topbar-btn-active" : ""}`}
            onClick={() => setShowUpload((v) => !v)}
            title={showUpload ? "Close upload panel" : "Upload PDF"}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            {showUpload ? "Close" : "Upload PDF"}
          </button>
          <div className="user-chip">
            <span className="user-avatar">{username?.[0]?.toUpperCase() || "U"}</span>
            <span className="user-name">{username}</span>
          </div>
          <button className="logout-btn" onClick={onLogout}>Sign Out</button>
        </div>
      </header>

      {/* Upload Panel */}
      {showUpload && (
        <div className="upload-panel-wrapper upload-panel-animate">
          <UploadArea
            chatId={chatId}
            token={token}
            onDocumentsChange={handleDocumentsChange}
            onUploadSuccess={handleUploadSuccess}
          />
        </div>
      )}

      {/* Messages */}
      <div className="messages-area">
        {loadingHistory && (
          <div className="history-loading">
            <div className="spinner-ring" /><span>Loading conversation…</span>
          </div>
        )}

        {!loadingHistory && messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <p>No messages yet.</p>
            <p className="empty-hint">
              {activeDocuments.length > 0
                ? "Documents attached. Ask anything about them below."
                : "Upload a PDF using the button above, then ask questions."}
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-wrapper">
        <form className="chat-input-area" onSubmit={sendMessage}>
          <textarea
            ref={(el) => { textareaRef.current = el; inputRef.current = el; }}
            className="chat-textarea"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents… (Shift+Enter for newline)"
            disabled={loading}
            rows={1}
            autoFocus
          />
          <button
            type="submit"
            className="send-btn"
            disabled={loading || !input.trim()}
            title="Send"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
        <p className="input-hint">Answers are grounded in your uploaded documents.</p>
      </div>
    </div>
  );
}
