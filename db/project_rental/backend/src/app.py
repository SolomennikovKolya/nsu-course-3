from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import jwt
import datetime
import uuid
from werkzeug.security import check_password_hash
import mysql.connector
import db.actions


SECRET_KEY = "access_secret"
REFRESH_SECRET_KEY = "refresh_secret"
ACCESS_EXPIRES_MIN = 15
REFRESH_EXPIRES_DAYS = 7


# app - экземпляр flask приложения
# CORS (Cross-Origin Resource Sharing) нужен, чтобы разрешить запросы с другого домена или порта
# "/api/*" ограничивает область действия CORS только на пути, начинающиеся с /api/
# "origins" указывает разрешённый источник — т.е. React-приложение, работающее на http://localhost:5173
# Уровени логирования: DEBUG, INFO, WARNING, ERROR, CRITICAL
app = Flask(__name__)
app.logger.setLevel("DEBUG")
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})


@app.cli.command("init_db")
def init_db():
    db.actions.init_db()


@app.cli.command("clear_db")
def clear_db():
    db.actions.clear_db()


@app.cli.command("seed_db")
def seed_db():
    db.actions.seed_db()


# Симуляция подключения к БД и получения пользователя
def get_user(identifier: str):
    conn = mysql.connector.connect(user='root', password='root', host='localhost', database='equipment_rental')
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT * FROM Users WHERE name = %s OR phone = %s OR email = %s
    """
    cursor.execute(query, (identifier, identifier, identifier))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user


# Хранилище refresh токенов (в реальности — Redis или таблица в БД)
refresh_store = {}


@app.route('/login', methods=['POST'])
def login():
    """
    Запрос на авторизацию (используется менеджерами и админами при входе).\n
    Request: Идентификатор пользователя (имя, телефон или емейл), пароль.\n
    Response: JWT токен и куки с Refresh токеном, если доступ разрешён.
    """
    app.logger.info(f"Incoming request: {request.method} {request.url}")
    app.logger.debug(f"Headers: {request.headers}")
    app.logger.debug(f"Body: {request.get_json()}")

    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')

    user = get_user(identifier)
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"msg": "Неверные данные"}), 401

    user_id = user['id']
    role = user['role']

    access_token = jwt.encode({
        "sub": user_id,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_EXPIRES_MIN)
    }, SECRET_KEY, algorithm="HS256")

    refresh_token = str(uuid.uuid4())
    refresh_store[refresh_token] = {
        "user_id": user_id,
        "role": role,
        "expires": datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_EXPIRES_DAYS)
    }

    response = make_response(jsonify({"access_token": access_token}))
    response.set_cookie("refresh_token", refresh_token, httponly=True, samesite='Strict')

    app.logger.debug(f"Outgoing response: {response.get_json()}")
    return response


@app.route('/refresh', methods=['POST'])
def refresh():
    """
    Запрос на обновление JWT токена по Refresh токену.
    Запрос будет успешным, если Refresh токен есть в базе, его срок действия не истёк.\n
    Request: Refresh токен.\n
    Response: JWT токен.
    """
    refresh_token = request.cookies.get('refresh_token')
    if not refresh_token or refresh_token not in refresh_store:
        return jsonify({"msg": "Invalid refresh token"}), 401

    record = refresh_store[refresh_token]
    if record['expires'] < datetime.datetime.utcnow():
        refresh_store.pop(refresh_token)
        return jsonify({"msg": "Refresh token expired"}), 401

    access_token = jwt.encode({
        "sub": record['user_id'],
        "role": record['role'],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_EXPIRES_MIN)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({"access_token": access_token})


@app.route('/logout', methods=['POST'])
def logout():
    """Выход из системы. Удаляется Refresh токен с сервера и куки с браузера."""
    refresh_token = request.cookies.get('refresh_token')
    if refresh_token:
        refresh_store.pop(refresh_token, None)
    response = make_response(jsonify({"msg": "Logged out"}))
    response.delete_cookie("refresh_token")
    return response


@app.route('/protected', methods=['GET'])
def protected():
    """Проверка JWT токена."""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"msg": "Missing auth header"}), 401

    try:
        token = auth_header.split()[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"msg": "Access granted", "role": payload['role']})
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Access token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Invalid token"}), 401


if __name__ == "__main__":
    app.run(debug=True)
