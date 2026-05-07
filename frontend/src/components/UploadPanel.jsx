import { useState, useRef } from "react";
import { upload } from "../services/api";

export default function UploadPanel({ token, onLogout }) {
  const [status, setStatus] = useState(null); // {type: 'success'|'error', text: string}
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef(null);

  const handleUpload = async () => {
    const file = fileRef.current?.files[0];
    if (!file) return;

    setStatus(null);
    setUploading(true);

    try {
      const result = await upload(file, token);
      setStatus({
        type: "success",
        text: `Uploaded! Document ID: ${result.document_id} — ${result.chunks} chunks ingested`,
      });
      fileRef.current.value = "";
    } catch (err) {
      if (err.message === "SESSION_EXPIRED") {
        onLogout();
        return;
      }
      setStatus({ type: "error", text: err.message });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-panel">
      <div className="upload-header">
        <span className="upload-icon">📄</span>
        <span>Upload PDF to Knowledge Base</span>
      </div>
      <div className="upload-controls">
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          disabled={uploading}
          className="upload-input"
        />
        <button
          className="upload-btn"
          onClick={handleUpload}
          disabled={uploading}
        >
          {uploading ? "Uploading…" : "Upload"}
        </button>
      </div>
      {status && (
        <div className={`upload-status upload-status-${status.type}`}>
          {status.text}
        </div>
      )}
    </div>
  );
}
