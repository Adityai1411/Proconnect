import React, { useEffect, useState } from 'react'
import { api } from '../api'
import { Bar, Pie } from 'react-chartjs-2'
import { Chart, ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js'
Chart.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend)

export default function Dashboard({ resumeId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!resumeId) return
    (async () => {
      try {
        const res = await api.get(`/resume/analysis/${resumeId}`)
        setData(res.data)
      } catch (e) {
        setError(e?.response?.data?.detail || e.message)
      }
    })()
  }, [resumeId])

  if (error) return <div className="card"><p>{error}</p></div>
  if (!data) return <div className="card"><p>Load a resume to see analysis.</p></div>

  const keywordCounts = data.keywords.slice(0, 12).reduce((acc, k) => {
    acc[k] = (acc[k] || 0) + 1; return acc
  }, {})

  const entities = data.entities || []
  const entityLabels = entities.map(e => e[1])
  const entityCounts = entityLabels.reduce((a,l)=>{a[l]=(a[l]||0)+1;return a},{})
  const entityKeys = Object.keys(entityCounts)

  return (
    <div className="card">
      <h2>Analysis for: {data.filename}</h2>
      <p><b>ATS-like score:</b> {data.score}/100</p>

      <div className="row">
        <div className="col">
          <h3>Top Keywords</h3>
          <Bar data={{
            labels: Object.keys(keywordCounts),
            datasets: [{ label: 'Frequency', data: Object.values(keywordCounts) }]
          }}/>
          <div style={{marginTop:8}}>
            {data.keywords.map(k => <span className="tag" key={k}>{k}</span>)}
          </div>
        </div>
        <div className="col">
          <h3>Entities by Type</h3>
          <Pie data={{
            labels: entityKeys,
            datasets: [{ data: entityKeys.map(k => entityCounts[k]) }]
          }}/>
          <details style={{marginTop:8}}>
            <summary>Entity Samples</summary>
            <pre>{JSON.stringify(entities.slice(0,30), null, 2)}</pre>
          </details>
        </div>
      </div>

      <details style={{marginTop:12}}>
        <summary><b>Sentence → Recommended Platform</b></summary>
        <ul>
          {Object.entries(data.sentence_to_platform).map(([s,p],i)=>(
            <li key={i}><b>[{p}]</b> {s}</li>
          ))}
        </ul>
      </details>

      <details style={{marginTop:12}}>
        <summary><b>Resume Text (first 10k chars)</b></summary>
        <pre>{(data.content || '').slice(0,10000)}</pre>
      </details>
    </div>
  )
}
