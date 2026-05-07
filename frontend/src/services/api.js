const API_URL = "http://localhost:8000";

/**
 * Authenticate and return a JWT token.
 * @param {string} username
 * @param {string} password
 * @returns {Promise<string>} access_token
 */
export async function login(username, password) {
  console.log("Calling:", `${API_URL}/login`);
  const res = await fetch(`${API_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (res.status === 401) {
    throw new Error("Invalid username or password");
  }
  if (res.status === 429) {
    throw new Error("Too many requests. Please wait and try again.");
  }
  if (!res.ok) {
    throw new Error(`Login failed (${res.status})`);
  }

  const data = await res.json();
  return data.access_token;
}

/**
 * Send a chat message and return the grounded response.
 * @param {string} message
 * @param {string} token
 * @param {string|null} conversationId
 * @returns {Promise<{answer: string, citations: string[], confidence: number, conversation_id: string|null}>}
 */
export async function chat(message, token, conversationId = null) {
  console.log("Calling:", `${API_URL}/chat`);
  const body = { message };
  if (conversationId) {
    body.conversation_id = conversationId;
  }

  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });

  if (res.status === 401) {
    throw new Error("SESSION_EXPIRED");
  }
  if (res.status === 429) {
    throw new Error("Too many requests. Please wait and try again.");
  }
  if (!res.ok) {
    throw new Error(`Chat request failed (${res.status})`);
  }

  return res.json();
}

/**
 * Upload a PDF file for ingestion into the knowledge base.
 * @param {File} file
 * @param {string} token
 * @returns {Promise<{document_id: string, chunks: number}>}
 */
export async function upload(file, token) {
  console.log("Calling:", `${API_URL}/upload`);
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/upload`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (res.status === 401) {
    throw new Error("SESSION_EXPIRED");
  }
  if (res.status === 403) {
    throw new Error("Admin privileges required");
  }
  if (res.status === 429) {
    throw new Error("Too many requests. Please wait and try again.");
  }
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Upload failed (${res.status})`);
  }

  return res.json();
}
