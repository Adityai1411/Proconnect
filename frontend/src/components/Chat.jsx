import React, { useState } from 'react'
import { api } from '../api'

export default function Chat({ resumeId }) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([]) // local display only
  const [loading, setLoading] = useState(false)

  const send = async () => {
    if (!input.trim()) return
    const userMsg = { role: 'user', content: input }
    setMessages(m => [...m, userMsg])
    setLoading(true)
    try {
      const res = await api.post('/chat', { message: input, resume_id: resumeId || null })
      const reply = res.data.response
      setMessages(m => [...m, { role: 'assistant', content: reply }])
    } catch (e) {
      setMessages(m => [...m, { role: 'assistant', content: `Error: ${e?.response?.data?.detail || e.message}` }])
    } finally {
      setInput(''); setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Chatbot</h2>
      <div style={{maxHeight: 320, overflowY: 'auto', padding: 8, border: '1px solid #27314d', borderRadius: 10}}>
        {messages.map((m, idx) => (
          <div key={idx} style={{margin: '8px 0'}}>
            <b>{m.role === 'user' ? 'You' : 'Assistant'}:</b> <span>{m.content}</span>
          </div>
        ))}
      </div>
      <div className="row" style={{marginTop: 10}}>
        <div className="col"><input value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask about your resume…" /></div>
        <div><button onClick={send} disabled={loading}>{loading ? 'Thinking…' : 'Send'}</button></div>
      </div>
    </div>
  )
}
