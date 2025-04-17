import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// document.getElementById('root') - Находит <div id="root"></div> в index.html
// ReactDOM.createRoot(...).render(...) -	Монтирует React-приложение в этот контейнер
// <StrictMode>	- Включает дополнительные проверки в dev-режиме (не обязателен, но полезен)
// <App /> - Рендерит корневой компонент приложения
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
