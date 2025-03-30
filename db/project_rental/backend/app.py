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
