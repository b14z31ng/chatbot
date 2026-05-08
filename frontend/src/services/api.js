const API_URL = "http://localhost:8000";

// ─── Auth ───────────────────────────────────────────────────────────────────

export async function login(username, password) {
  const res = await fetch(`${API_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (res.status === 401) throw new Error("Invalid username or password");
  if (res.status === 429) throw new Error("Too many requests. Please wait and try again.");
  if (!res.ok) throw new Error(`Login failed (${res.status})`);
  const data = await res.json();
  return data.access_token;
}

// ─── Chats ──────────────────────────────────────────────────────────────────

export async function getChats(token) {
  const res = await _authFetch(`${API_URL}/chats`, { method: "GET" }, token);
  return res.json();
}

export async function createChat(token, title = null) {
  const res = await _authFetch(`${API_URL}/chats`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }, token);
  return res.json();
}

export async function deleteChat(chatId, token) {
  await _authFetch(`${API_URL}/chats/${chatId}`, { method: "DELETE" }, token);
}

export async function renameChat(chatId, title, token) {
  const res = await _authFetch(`${API_URL}/chats/${chatId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }, token);
  return res.json();
}

// ─── Messages ───────────────────────────────────────────────────────────────

export async function getChatMessages(chatId, token) {
  const res = await _authFetch(`${API_URL}/chats/${chatId}/messages`, { method: "GET" }, token);
  return res.json();
}

export async function sendChatMessage(chatId, message, token) {
  const res = await _authFetch(`${API_URL}/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  }, token);
  return res.json();
}

// ─── Documents ──────────────────────────────────────────────────────────────

export async function uploadToChat(chatId, file, token) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await _authFetch(`${API_URL}/chats/${chatId}/upload`, {
    method: "POST",
    body: formData,
  }, token);
  return res.json();
}

export async function getChatDocuments(chatId, token) {
  const res = await _authFetch(`${API_URL}/chats/${chatId}/documents`, { method: "GET" }, token);
  return res.json();
}

export async function deleteDocument(docId, token) {
  await _authFetch(`${API_URL}/documents/${docId}`, { method: "DELETE" }, token);
}

// ─── Legacy endpoints (kept for backwards compat) ────────────────────────────

export async function chat(message, token, conversationId = null) {
  const body = { message };
  if (conversationId) body.conversation_id = conversationId;
  const res = await _authFetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, token);
  return res.json();
}

export async function upload(file, token) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await _authFetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  }, token);
  return res.json();
}

// ─── Internal fetch helper ───────────────────────────────────────────────────

async function _authFetch(url, options, token) {
  const headers = {
    ...(options.headers || {}),
    Authorization: `Bearer ${token}`,
  };
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) throw new Error("SESSION_EXPIRED");
  if (res.status === 403) throw new Error("Access denied");
  if (res.status === 429) throw new Error("Too many requests. Please wait and try again.");
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return res;
}
