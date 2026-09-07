import axios from 'axios';

// ─────────────────────────────────────────────────────────
// Axios client — swap VITE_API_BASE_URL env var to connect
// to the real FastAPI backend (ml/serving/inference_server.py)
// ─────────────────────────────────────────────────────────
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 300000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// Request interceptor — attach auth token if present
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('omnisentinel_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Response interceptor — normalize errors
client.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      (err.code === 'ECONNABORTED' ? 'Request timed out. Is the API server running?' : 'Unable to reach the server. Running in demo mode.') ||
      err.message;
    return Promise.reject(new Error(message));
  }
);

export default client;
