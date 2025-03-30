
**Начальная настройка:** 
- `cd ../backend && .venv\Scripts\activate` - активация виртуального окружения
- `pip install -r requirements.txt` - установка зависимостей

**Development:**
- `flask run` (из папки backend) - запуск бэкенда (порт 5000)
- `npm start` (из папки frontend) - запуск фронта (порт 3000). Благодаря этому:
	- Запускается webpack dev server
	- Компиляция React-приложения происходит в памяти (без создания файлов)
	- Включается Hot Module Replacement (HMR) — автоматическое обновление кода без перезагрузки страницы

**Production:**
- `npm run build` (из папки frontend) - сборка фронта
- Вариант A: Копируете build в `backend/static` и используете Flask для статики
- Вариант B: Настраиваете веб-сервер (Nginx) для раздельного обслуживания