import { useState, useEffect } from "react";
import LoginForm from "./components/LoginForm";
import ChatBox from "./components/ChatBox";
import "./App.css";

const TOKEN_KEY = "rag_chat_token";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));

  const handleLogin = (newToken) => {
    localStorage.setItem(TOKEN_KEY, newToken);
    setToken(newToken);
  };

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
  };

  if (!token) {
    return <LoginForm onLogin={handleLogin} />;
  }

  return <ChatBox token={token} onLogout={handleLogout} />;
}
