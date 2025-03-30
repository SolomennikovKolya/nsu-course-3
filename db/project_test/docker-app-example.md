
## Архитектура приложения

Приложение будет состоять из:
1. Бэкенд на Flask (Python)
2. Фронтенд на React (JavaScript)
3. База данных MySQL

## Шаг 1: Настройка проекта

### 1.1 Структура проекта

```
myapp/
├── backend/              # Flask бэкенд
│   ├── app.py            # Основное приложение
│   ├── auth.py           # Аутентификация
│   ├── queries/          # SQL-запросы
│   │   ├── admin.sql
│   │   ├── manager.sql
│   │   └── user.sql
│   ├── requirements.txt  # Зависимости
│   └── config.py         # Конфигурация
├── frontend/             # React фронтенд
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   │   ├── Admin.js
│   │   │   ├── Manager.js
│   │   │   └── User.js
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── ...
├── docker-compose.yml    # Для развертывания
└── README.md
```

### 1.2 Установка зависимостей

Для бэкенда (`backend/requirements.txt`):
```
flask
flask-mysqldb
flask-cors
python-dotenv
```

## Шаг 2: Настройка базы данных MySQL

### 2.1 Создание базы данных и пользователей

```sql
-- Создание базы данных
CREATE DATABASE app_db;

-- Создание ролей и пользователей
CREATE USER 'app_admin'@'%' IDENTIFIED BY 'admin_password';
CREATE USER 'app_manager'@'%' IDENTIFIED BY 'manager_password';
CREATE USER 'app_user'@'%' IDENTIFIED BY 'user_password';

-- Назначение прав
GRANT ALL PRIVILEGES ON app_db.* TO 'app_admin'@'%';
GRANT SELECT, INSERT, UPDATE ON app_db.* TO 'app_manager'@'%';
GRANT SELECT ON app_db.* TO 'app_user'@'%';

FLUSH PRIVILEGES;

-- Создание таблицы пользователей
USE app_db;
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role ENUM('admin', 'manager', 'user') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Шаг 3: Реализация бэкенда на Flask

### 3.1 Основное приложение (`backend/app.py`)

```python
from flask import Flask, jsonify, request
from auth import authenticate_user
from flask_mysqldb import MySQL
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Конфигурация MySQL
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'app_db')
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = authenticate_user(mysql, username, password)
    
    if user:
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role']
            }
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/admin/data', methods=['GET'])
def admin_data():
    # Проверка роли администратора
    user = request.user  # Должно быть добавлено middleware для проверки аутентификации
    
    if user['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Загрузка SQL запроса из файла
    with open('queries/admin.sql', 'r') as file:
        sql_query = file.read()
    
    cursor = mysql.connection.cursor()
    cursor.execute(sql_query)
    data = cursor.fetchall()
    cursor.close()
    
    return jsonify(data)

# Аналогичные маршруты для manager и user

if __name__ == '__main__':
    app.run(debug=True)
```

### 3.2 Аутентификация (`backend/auth.py`)

```python
def authenticate_user(mysql, username, password):
    cursor = mysql.connection.cursor()
    
    # Используем встроенные средства MySQL для проверки пароля
    query = "SELECT id, username, role FROM users WHERE username = %s AND password = SHA2(%s, 256)"
    cursor.execute(query, (username, password))
    
    user = cursor.fetchone()
    cursor.close()
    
    return user
```

## Шаг 4: Реализация фронтенда на React

### 4.1 Основной компонент (`frontend/src/App.js`)

```jsx
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import AdminPage from './pages/Admin';
import ManagerPage from './pages/Manager';
import UserPage from './pages/User';
import Login from './components/Login';

function App() {
  const [user, setUser] = useState(null);

  const handleLogin = async (username, password) => {
    const response = await fetch('http://localhost:5000/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await response.json();
    if (data.success) {
      setUser(data.user);
      return true;
    }
    return false;
  };

  const handleLogout = () => {
    setUser(null);
  };

  return (
    <Router>
      <div className="App">
        {!user ? (
          <Login onLogin={handleLogin} />
        ) : (
          <>
            <button onClick={handleLogout}>Logout</button>
            <Routes>
              <Route path="/admin" element={
                user.role === 'admin' ? <AdminPage user={user} /> : <Navigate to={`/${user.role}`} />
              } />
              <Route path="/manager" element={
                user.role === 'manager' ? <ManagerPage user={user} /> : <Navigate to={`/${user.role}`} />
              } />
              <Route path="/user" element={<UserPage user={user} />} />
              <Route path="*" element={<Navigate to={`/${user.role}`} />} />
            </Routes>
          </>
        )}
      </div>
    </Router>
  );
}

export default App;
```

### 4.2 Страница администратора (`frontend/src/pages/Admin.js`)

```jsx
import React, { useEffect, useState } from 'react';

function AdminPage({ user }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('http://localhost:5000/admin/data', {
        headers: { 'Authorization': `Bearer ${user.id}` }
      });
      
      if (response.ok) {
        const result = await response.json();
        setData(result);
      }
    };
    
    fetchData();
  }, [user]);

  return (
    <div>
      <h1>Admin Dashboard</h1>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
          </tr>
        </thead>
        <tbody>
          {data.map(item => (
            <tr key={item.id}>
              <td>{item.id}</td>
              <td>{item.username}</td>
              <td>{item.role}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default AdminPage;
```

## Шаг 5: Развертывание приложения

### 5.1 Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - MYSQL_HOST=db
      - MYSQL_USER=app_admin
      - MYSQL_PASSWORD=admin_password
      - MYSQL_DB=app_db
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  db:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=root_password
      - MYSQL_DATABASE=app_db
      - MYSQL_USER=app_admin
      - MYSQL_PASSWORD=admin_password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

### 5.2 Dockerfile для бэкенда (`backend/Dockerfile`)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### 5.3 Dockerfile для фронтенда (`frontend/Dockerfile`)

```dockerfile
FROM node:16

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

CMD ["npm", "start"]
```

## Шаг 6: Запуск приложения

1. Убедитесь, что у вас установлены Docker и Docker Compose
2. В корне проекта выполните команду:
   ```bash
   docker-compose up --build
   ```
3. Приложение будет доступно:
   - Фронтенд: http://localhost:3000
   - Бэкенд: http://localhost:5000
   - MySQL: localhost:3306

## Дополнительные улучшения

1. **Безопасность**:
   - Добавьте HTTPS
   - Реализуйте JWT для аутентификации
   - Добавьте CSRF защиту

2. **Производительность**:
   - Добавьте кэширование запросов
   - Оптимизируйте SQL-запросы

3. **Масштабируемость**:
   - Добавьте балансировщик нагрузки
   - Реализуйте репликацию базы данных

Это базовый каркас приложения, который можно расширять в зависимости от конкретных требований.