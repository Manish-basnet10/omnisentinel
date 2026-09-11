import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  envDir: '.', // Stops Vite from scanning the huge parent folder for .env files
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/predict': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/demo-stream': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/model-info': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
