import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// plugins - Сюда подключаются плагины для поддержки фреймворков
// Проксирует запросы на Flask API, чтобы писать например /api/auth/login вместо http://localhost:5000/api/auth/login
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})