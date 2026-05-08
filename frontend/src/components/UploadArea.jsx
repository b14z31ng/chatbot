import { useState, useRef, useEffect, useCallback } from "react";
import { uploadToChat, getChatDocuments, deleteDocument } from "../services/api";
import { toast } from "./Toast";

export default function UploadArea({ chatId, token, onDocumentsChange, onUploadSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileRef = useRef(null);

  // ── Load documents only when chatId changes (NOT on every render) ─────────
  const loadDocuments = useCallback(async () => {
    if (!chatId) return;
    try {
      const docs = await getChatDocuments(chatId, token);
      setDocuments(docs);
      onDocumentsChange?.(docs);
    } catch (err) {
      if (err.message !== "SESSION_EXPIRED") {
        toast("Failed to load documents", "error");
      }
    }
  }, [chatId, token]); // intentionally exclude onDocumentsChange to avoid loop

  useEffect(() => {
    setDocuments([]); // clear stale docs on chat switch
    loadDocuments();
  }, [loadDocuments]); // fires only when chatId/token change

  const handleFile = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast("Only PDF files are supported", "error");
      return;
    }
    if (file.size > 10_000_000) {
      toast("File too large (max 10MB)", "error");
      return;
    }
    if (uploading) return; // prevent double upload

    setUploading(true);
    setProgress(10);

    const progressInterval = setInterval(() => {
      setProgress((p) => (p < 85 ? p + 5 : p));
    }, 300);

    try {
      const doc = await uploadToChat(chatId, file, token);
      clearInterval(progressInterval);
      setProgress(100);
      setTimeout(() => setProgress(0), 600);

      // Update local state directly — no re-fetch needed
      setDocuments((prev) => [...prev.filter((d) => d.id !== doc.id), doc]);
      onDocumentsChange?.([...documents.filter((d) => d.id !== doc.id), doc]);
      toast(`"${doc.filename}" uploaded — ready to query`, "success");
      onUploadSuccess?.();   // signal ChatBox to auto-close after success
    } catch (err) {
      clearInterval(progressInterval);
      setProgress(0);
      if (err.message !== "SESSION_EXPIRED") {
        toast(err.message || "Upload failed", "error");
      }
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleDelete = async (docId, filename) => {
    try {
      await deleteDocument(docId, token);
      const updated = documents.filter((d) => d.id !== docId);
      setDocuments(updated);
      onDocumentsChange?.(updated);
      toast(`"${filename}" removed`, "info");
    } catch (err) {
      if (err.message !== "SESSION_EXPIRED") {
        toast("Failed to remove document", "error");
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="upload-area">
      <div
        className={`drop-zone ${dragging ? "drop-zone-active" : ""} ${uploading ? "drop-zone-uploading" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files?.[0])}
          disabled={uploading}
        />
        <div className="drop-zone-icon">
          {uploading ? (
            <div className="upload-spinner" />
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          )}
        </div>
        <p className="drop-zone-text">
          {uploading ? "Uploading…" : dragging ? "Drop PDF here" : "Upload PDF"}
        </p>
        <p className="drop-zone-hint">Click or drag & drop · Max 10MB</p>

        {progress > 0 && (
          <div className="upload-progress">
            <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
          </div>
        )}
      </div>

      {documents.length > 0 && (
        <div className="doc-list">
          {documents.map((doc) => (
            <div key={doc.id} className="doc-item">
              <span className="doc-icon">📄</span>
              <span className="doc-name" title={doc.filename}>{doc.filename}</span>
              <button
                className="doc-delete-btn"
                onClick={() => handleDelete(doc.id, doc.filename)}
                title="Remove document"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
