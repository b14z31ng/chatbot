import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

function formatTime(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

function ConfidenceBadge({ confidence }) {
  if (confidence === undefined || confidence === null) return null;
  const pct = Math.round(confidence);
  const cls =
    pct >= 70 ? "confidence-high" : pct >= 40 ? "confidence-mid" : "confidence-low";
  return <span className={`confidence-badge ${cls}`}>{pct}% confidence</span>;
}

export default function MessageBubble({ message }) {
  const { role, content, confidence, citations, created_at } = message;

  if (role === "error") {
    return (
      <div className="message message-error">
        <div className="message-bubble">
          <p className="message-text">⚠ {content}</p>
        </div>
      </div>
    );
  }

  const isUser = role === "user";

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      {!isUser && (
        <div className="msg-avatar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2a7 7 0 00-7 7c0 2.4 1.2 4.5 3 5.7V17a2 2 0 002 2h4a2 2 0 002-2v-2.3c1.8-1.2 3-3.3 3-5.7a7 7 0 00-7-7z" />
          </svg>
        </div>
      )}

      <div className="message-bubble">
        {isUser ? (
          <p className="message-text">{content}</p>
        ) : (
          <div className="message-markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || "");
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={`inline-code ${className || ""}`} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && (
          <div className="message-meta">
            {citations && citations.length > 0 && (
              <div className="citations">
                <span className="meta-label">Sources:</span>
                {citations.map((c, j) => (
                  <span key={j} className="citation-tag">{c}</span>
                ))}
              </div>
            )}
            <div className="meta-right">
              <ConfidenceBadge confidence={confidence} />
              {created_at && (
                <span className="msg-time">{formatTime(created_at)}</span>
              )}
            </div>
          </div>
        )}

        {isUser && created_at && (
          <span className="msg-time msg-time-user">{formatTime(created_at)}</span>
        )}
      </div>
    </div>
  );
}
