import React, { useState } from 'react'
import { api } from '../api'

export default function Upload({ onUploaded }) {
  const [file, setFile] = useState(null)
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState('')

  const submit = async () => {
    if (!file) return
    setLoading(true)
    setMsg('')
    try {
      const form = new FormData()
      form.append('file', file)
      if (email) form.append('user_email', email)

      console.log("Uploading file:", file.name) // Debug
      const res = await api.post('/resume/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      console.log("Upload response:", res.data) // Debug
      onUploaded(res.data.resume_id)
      setMsg('Uploaded & analyzed successfully ✅')
    } catch (e) {
      console.error("Upload error:", e)
      setMsg(`Error: ${e?.response?.data?.detail || e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Upload Resume</h2>
      <div className="row">
        <div className="col">
          <input type="file" onChange={e => setFile(e.target.files[0])}/>
        </div>
        <div className="col">
          <input type="email" placeholder="(Optional) Email to create user"
                 value={email} onChange={e=>setEmail(e.target.value)} />
        </div>
      </div>
      <br/>
      <button onClick={submit} disabled={loading || !file}>
        {loading ? 'Processing…' : 'Upload & Analyze'}
      </button>
      {msg && <p>{msg}</p>}
    </div>
  )
}
