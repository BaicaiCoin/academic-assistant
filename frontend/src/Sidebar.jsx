import { PlusIcon, MessageSquareIcon } from 'lucide-react'

export default function Sidebar({ threads, activeThread, onSelect, onNew }) {
  return (
    <aside
      className="flex flex-col h-full"
      style={{
        width: 'var(--sidebar-w)',
        background: '#0a0a0a',
        borderRight: '1px solid #1e1e1e',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div className="p-4" style={{ borderBottom: '1px solid #1e1e1e' }}>
        <div className="flex items-center gap-2 mb-4">
          <span
            className="text-xs font-mono tracking-widest uppercase"
            style={{ color: '#a89b7a' }}
          >
            Academic
          </span>
          <span
            className="text-xs font-mono tracking-widest uppercase"
            style={{ color: '#555' }}
          >
            Assistant
          </span>
        </div>

        <button
          onClick={onNew}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors"
          style={{ background: '#1a1a18', border: '1px solid #2a2a28', color: '#e8e4dc' }}
          onMouseEnter={e => e.currentTarget.style.borderColor = '#a89b7a'}
          onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2a28'}
        >
          <PlusIcon size={14} />
          新建会话
        </button>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto p-2">
        {threads.length === 0 && (
          <p className="text-xs text-center mt-8 font-mono" style={{ color: '#444' }}>
            暂无会话记录
          </p>
        )}
        {threads.map(t => (
          <button
            key={t.id}
            onClick={() => onSelect(t.id)}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-left text-sm mb-1 transition-all"
            style={{
              background: activeThread === t.id ? '#1e1c17' : 'transparent',
              border: `1px solid ${activeThread === t.id ? '#3d3325' : 'transparent'}`,
              color: activeThread === t.id ? '#f0ece4' : '#888',
            }}
            onMouseEnter={e => {
              if (activeThread !== t.id) e.currentTarget.style.color = '#ccc'
            }}
            onMouseLeave={e => {
              if (activeThread !== t.id) e.currentTarget.style.color = '#888'
            }}
          >
            <MessageSquareIcon size={13} style={{ flexShrink: 0, opacity: 0.6 }} />
            <span className="truncate font-mono text-xs">{t.title || t.id.slice(0, 16) + '…'}</span>
          </button>
        ))}
      </div>

      {/* Footer */}
      <div
        className="p-4 text-xs font-mono"
        style={{ borderTop: '1px solid #1e1e1e', color: '#333' }}
      >
        v0.1.0 · GraphRAG
      </div>
    </aside>
  )
}
