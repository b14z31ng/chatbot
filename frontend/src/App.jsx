import { useState, useEffect, useCallback } from "react";
import LoginForm from "./components/LoginForm";
import ChatBox from "./components/ChatBox";
import Sidebar from "./components/Sidebar";
import Toast, { useToast } from "./components/Toast";
import { getChats, createChat } from "./services/api";
import "./App.css";

const TOKEN_KEY = "rag_chat_token";
const USER_KEY = "rag_chat_user";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [username, setUsername] = useState(() => localStorage.getItem(USER_KEY) || "admin");
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { toasts, addToast, removeToast } = useToast();

  const handleLogin = (newToken, user = "admin") => {
    localStorage.setItem(TOKEN_KEY, newToken);
    localStorage.setItem(USER_KEY, user);
    setToken(newToken);
    setUsername(user);
  };

  const handleLogout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setToken(null);
    setChats([]);
    setActiveChatId(null);
  }, []);

  // Load chat list
  const loadChats = useCallback(async () => {
    if (!token) return;
    try {
      const data = await getChats(token);
      setChats(data);
      // Auto-select first chat or keep current
      setActiveChatId((prev) => prev ?? data[0]?.id ?? null);
    } catch (err) {
      if (err.message === "SESSION_EXPIRED") handleLogout();
    }
  }, [token, handleLogout]);

  useEffect(() => {
    loadChats();
  }, [loadChats]);

  const handleChatUpdated = useCallback(() => {
    loadChats();
  }, [loadChats]);

  const activeChat = chats.find((c) => c.id === activeChatId);

  if (!token) {
    return (
      <>
        <LoginForm onLogin={handleLogin} />
        <Toast toasts={toasts} onRemove={removeToast} />
      </>
    );
  }

  return (
    <div className={`app-layout ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
      {/* Mobile sidebar toggle */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        token={token}
        onSelectChat={(id) => setActiveChatId(id)}
        onChatsChange={setChats}
        onNewChat={handleChatUpdated}
      />

      <main className="main-area">
        <ChatBox
          chatId={activeChatId}
          chatTitle={activeChat?.title}
          token={token}
          username={username}
          onLogout={handleLogout}
          onChatUpdated={handleChatUpdated}
        />
      </main>

      <Toast toasts={toasts} onRemove={removeToast} />
    </div>
  );
}
