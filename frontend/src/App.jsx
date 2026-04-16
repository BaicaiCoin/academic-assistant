import { useState, useRef, useEffect, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { SendIcon, SquareIcon } from 'lucide-react'
import Sidebar from './Sidebar'
import MessageBubble from './MessageBubble'
import { sendMessage } from './api'

function loadThreads() {
  try { return JSON.parse(localStorage.getItem('threads') || '[]') } catch { return [] }
}
function saveThreads(threads) {
  localStorage.setItem('threads', JSON.stringify(threads))
}
function loadMessages(threadId) {
  try { return JSON.parse(localStorage.getItem(`msgs_${threadId}`) || '[]') } catch { return [] }
}
function saveMessages(threadId, messages) {
  localStorage.setItem(`msgs_${threadId}`, JSON.stringify(messages))
}

export default function App() {
  const [threads, setThreads] = useState(loadThreads)
  const [activeThreadId, setActiveThreadId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const bottomRef = useRef(null)
  const abortRef = useRef(false)

  useEffect(() => {
    if (activeThreadId) setMessages(loadMessages(activeThreadId))
  }, [activeThreadId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (activeThreadId && messages.length > 0) saveMessages(activeThreadId, messages)
  }, [messages, activeThreadId])

  const createThread = useCallback(() => {
    const id = uuidv4()
    const thread = { id, title: '新会话', createdAt: Date.now() }
    const updated = [thread, ...threads]
    setThreads(updated)
    saveThreads(updated)
    setActiveThreadId(id)
    setMessages([])
    return id
  }, [threads])

  const updateThreadTitle = useCallback((threadId, title) => {
    setThreads(prev => {
      const updated = prev.map(t => t.id === threadId ? { ...t, title } : t)
      saveThreads(updated)
      return updated
    })
  }, [])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || isStreaming) return

    let threadId = activeThreadId
    if (!threadId) threadId = createThread()

    setInput('')
    setIsStreaming(true)
    abortRef.current = false

    const userMsg = { id: uuidv4(), role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])

    const assistantId = uuidv4()
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '', status: '' }])

    const isFirstMessage = loadMessages(threadId).length === 0
    if (isFirstMessage) {
      updateThreadTitle(threadId, text.slice(0, 30) + (text.length > 30 ? '...' : ''))
    }

    await sendMessage(threadId, text, {
      onStatus: (msg) => {
        if (abortRef.current) return
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, status: msg } : m))
      },
      onAnswer: (answer) => {
        if (abortRef.current) return
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: answer, status: '' } : m))
      },
      onDone: () => setIsStreaming(false),
      onError: (err) => {
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: `错误：${err}`, status: '' } : m
        ))
        setIsStreaming(false)
      },
    })
  }, [input, isStreaming, activeThreadId, createThread, updateThreadTitle])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="flex h-screen">
      <Sidebar threads={threads} activeThread={activeThreadId} onSelect={setActiveThreadId} onNew={createThread} />

      <div className="flex flex-col flex-1 min-w-0">
        <div className="flex-1 overflow-y-auto px-6 py-8" style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-3 select-none">
              <div className="w-12 h-12 rounded-full flex items-center justify-center text-xl font-mono"
                style={{ background: '#1a1a18', border: '1px solid #3d3325', color: '#a89b7a' }}>A</div>
              <p className="text-sm font-mono" style={{ color: '#555' }}>提问关于论文或视频的任何内容</p>
            </div>
          )}
          {messages.map(msg => (
            <MessageBubble key={msg.id} role={msg.role} content={msg.content} status={msg.status} />
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="px-6 pb-6" style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
          <div className="flex items-end gap-3 rounded-2xl px-4 py-3"
            style={{ background: '#141412', border: '1px solid #2a2a28' }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题，Enter 发送，Shift+Enter 换行..."
              rows={1}
              disabled={isStreaming}
              className="flex-1 bg-transparent resize-none outline-none text-sm leading-relaxed"
              style={{ color: '#e8e4dc', minHeight: '24px', maxHeight: '160px', fontFamily: 'inherit', caretColor: '#a89b7a' }}
              onInput={e => {
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
              }}
            />
            <button
              onClick={isStreaming ? () => { abortRef.current = true; setIsStreaming(false) } : handleSend}
              disabled={!isStreaming && !input.trim()}
              className="flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all"
              style={{
                background: isStreaming ? '#2a1a1a' : (input.trim() ? '#2a2318' : '#111'),
                border: `1px solid ${isStreaming ? '#7a3a3a' : (input.trim() ? '#a89b7a' : '#222')}`,
                color: isStreaming ? '#e87070' : (input.trim() ? '#a89b7a' : '#444'),
                cursor: (!isStreaming && !input.trim()) ? 'not-allowed' : 'pointer',
              }}
            >
              {isStreaming ? <SquareIcon size={14} /> : <SendIcon size={14} />}
            </button>
          </div>
          <p className="text-center text-xs mt-2 font-mono" style={{ color: '#333' }}>
            由 DeepSeek + GraphRAG 驱动
          </p>
        </div>
      </div>
    </div>
  )
}
