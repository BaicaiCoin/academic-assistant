import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

export default function MessageBubble({ role, content, status }) {
  const isUser = role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div
          className="max-w-[75%] px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed"
          style={{ background: '#2a2318', border: '1px solid #3d3325', color: '#f0ece4' }}
        >
          {content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 mb-6">
      {/* Avatar */}
      <div
        className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono mt-1"
        style={{ background: '#2a2318', border: '1px solid #a89b7a', color: '#a89b7a' }}
      >
        A
      </div>

      <div className="flex-1 min-w-0">
        {/* Status badge */}
        {status && (
          <div className="flex items-center gap-2 mb-2">
            <span className="pulse-dot w-1.5 h-1.5 rounded-full bg-amber-500 inline-block" />
            <span className="text-xs font-mono" style={{ color: '#a89b7a' }}>{status}</span>
          </div>
        )}

        {/* Answer */}
        {content && (
          <div
            className="prose text-sm rounded-2xl rounded-tl-sm px-4 py-3"
            style={{ background: '#1a1a18', border: '1px solid #2a2a28', color: '#e8e4dc', maxWidth: '100%' }}
          >
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}

        {/* Thinking indicator (no content yet) */}
        {!content && !status && (
          <div className="flex gap-1 px-4 py-3">
            {[0, 1, 2].map(i => (
              <span
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-amber-600"
                style={{ animation: `pulse-dot 1.2s ease-in-out ${i * 0.2}s infinite` }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
