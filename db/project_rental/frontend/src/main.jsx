import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './main.css';

// main.jsx - Точка входа для React-приложения
// document.getElementById('root') - Находит элемент <div id="root"></div> в index.html
// createRoot(...) - Создаёт root (точку монтирования) React внутри этого элемента
// render(...) - Запускает первую отрисовку
// Монтирование - Это процесс подключения компонента к DOM-дереву браузера
// <StrictMode>	- Включает дополнительные проверки в dev-режиме (не обязателен, но полезен)
// <App /> - Рендерит корневой компонент приложения
createRoot(document.getElementById('root')).render(
  // <StrictMode>
  <App />
  // </StrictMode>,
)
