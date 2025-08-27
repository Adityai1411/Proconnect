import React, { useState } from 'react'
import Upload from './components/Upload.jsx'
import Dashboard from './components/Dashboard.jsx'
import Chat from './components/Chat.jsx'

export default function App() {
  const [resumeId, setResumeId] = useState(null)

  return (
    <div className="container">
      <h1>📄 Proconnect</h1>
      <p>Upload your resume → analyze → visualize → chat with context.</p>
      <Upload onUploaded={setResumeId} />
      <div className="row">
        <div className="col"><Dashboard resumeId={resumeId} /></div>
        <div className="col"><Chat resumeId={resumeId} /></div>
      </div>
    </div>
  )
}
