import axios from 'axios'

// adjust if backend runs elsewhere
export const API_BASE = 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE
})
