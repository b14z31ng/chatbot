import { useState, memo } from "react";
import { deleteChat, renameChat, createChat } from "../services/api";
import { toast } from "./Toast";

function formatRelativeDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now - d;
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  return d.toLocaleDateString();
}

const ChatItem = memo(function ChatItem({ chat, isActive, token, onSelect, onDelete, onRename }) {
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(chat.title);
  const [showActions, setShowActions] = useState(false);

  const handleRenameSubmit = async (e) => {
    e.preventDefault();
    if (!editTitle.trim() || editTitle === chat.title) {
      setEditing(false);
      return;
    }
    try {
      await renameChat(chat.id, editTitle.trim(), token);
      onRename(chat.id, editTitle.trim());
      toast("Chat renamed", "success");
    } catch {
      toast("Failed to rename chat", "error");
    }
    setEditing(false);
  };

  const handleDelete = async (e) => {
    e.stopPropagation();
    try {
      await deleteChat(chat.id, token);
      onDelete(chat.id);
      toast("Chat deleted", "info");
    } catch {
      toast("Failed to delete chat", "error");
    }
  };

  return (
    <div
      className={`chat-item ${isActive ? "chat-item-active" : ""}`}
      onClick={() => !editing && onSelect(chat.id)}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className="chat-item-icon">💬</div>
      <div className="chat-item-body">
        {editing ? (
          <form onSubmit={handleRenameSubmit} onClick={(e) => e.stopPropagation()}>
            <input
              className="chat-rename-input"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              autoFocus
              onBlur={handleRenameSubmit}
              maxLength={40}
            />
          </form>
        ) : (
          <>
            <span className="chat-item-title">{chat.title}</span>
            <span className="chat-item-date">{formatRelativeDate(chat.updated_at)}</span>
          </>
        )}
      </div>

      {(showActions || isActive) && !editing && (
        <div className="chat-item-actions" onClick={(e) => e.stopPropagation()}>
          <button
            className="chat-action-btn"
            onClick={() => { setEditing(true); setEditTitle(chat.title); }}
            title="Rename"
          >
            ✏
          </button>
          <button
            className="chat-action-btn chat-action-delete"
            onClick={handleDelete}
            title="Delete"
          >
            🗑
          </button>
        </div>
      )}

      {chat.document_count > 0 && (
        <span className="chat-doc-badge" title={`${chat.document_count} file(s)`}>
          📎 {chat.document_count}
        </span>
      )}
    </div>
  );
});

export default function Sidebar({ chats, activeChatId, token, onSelectChat, onChatsChange, onNewChat }) {
  const handleNewChat = async () => {
    try {
      const newChat = await createChat(token);
      onChatsChange([newChat, ...chats]);
      onSelectChat(newChat.id);
    } catch {
      toast("Failed to create chat", "error");
    }
  };

  const handleDelete = (chatId) => {
    const updated = chats.filter((c) => c.id !== chatId);
    onChatsChange(updated);
    if (activeChatId === chatId) {
      onSelectChat(updated[0]?.id ?? null);
    }
  };

  const handleRename = (chatId, newTitle) => {
    onChatsChange(chats.map((c) => c.id === chatId ? { ...c, title: newTitle } : c));
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <span className="sidebar-logo-icon">✦</span>
          <span className="sidebar-logo-text">RAG Chat</span>
        </div>
        <button className="new-chat-btn" onClick={handleNewChat} title="New Chat">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Chat
        </button>
      </div>

      <div className="sidebar-section-label">Conversations</div>

      <div className="chat-list">
        {chats.length === 0 && (
          <div className="sidebar-empty">
            <p>No conversations yet.</p>
            <p>Click "New Chat" to start!</p>
          </div>
        )}
        {chats.map((chat) => (
          <ChatItem
            key={chat.id}
            chat={chat}
            isActive={chat.id === activeChatId}
            token={token}
            onSelect={onSelectChat}
            onDelete={handleDelete}
            onRename={handleRename}
          />
        ))}
      </div>

      <div className="sidebar-footer">
        <span className="sidebar-footer-text">Build By Reshad</span>
      </div>
    </aside>
  );
}
