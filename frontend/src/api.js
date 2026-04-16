const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Send a message and stream back SSE events.
 * Calls onStatus(msg), onAnswer(text), onDone(), onError(msg).
 */
export async function sendMessage(threadId, message, { onStatus, onAnswer, onDone, onError }) {
  const res = await fetch(`${API_BASE}/chat/${threadId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })

  if (!res.ok) {
    onError?.(`HTTP ${res.status}`)
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const event = JSON.parse(line.slice(6))
        if (event.type === 'status') onStatus?.(event.content)
        else if (event.type === 'answer') onAnswer?.(event.content)
        else if (event.type === 'done') onDone?.()
        else if (event.type === 'error') onError?.(event.content)
      } catch (_) {}
    }
  }
}

export async function listThreads() {
  const res = await fetch(`${API_BASE}/threads`)
  if (!res.ok) return []
  const data = await res.json()
  return data.threads || []
}
