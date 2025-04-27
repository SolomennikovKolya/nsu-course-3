from flask import Blueprint, request, jsonify, make_response
import jwt
from datetime import datetime, timezone, timedelta
import uuid
from werkzeug.security import check_password_hash, generate_password_hash
from pathlib import Path
from dotenv import load_dotenv
import os
import db.actions


general_bp = Blueprint('general', __name__, url_prefix='/')

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')
SECRET_KEY = os.getenv('SECRET_KEY')
REFRESH_SECRET_KEY = os.getenv('REFRESH_SECRET_KEY')
ACCESS_EXPIRES_MIN = int(os.getenv('ACCESS_EXPIRES_MIN'))
REFRESH_EXPIRES_DAYS = int(os.getenv('REFRESH_EXPIRES_DAYS'))


@general_bp.route('/login', methods=['POST'])
def login():
    """
    Запрос на авторизацию (используется менеджерами и админами при входе).\n
    Request: Идентификатор пользователя (имя, телефон или емейл), пароль.\n
    Response: JWT токен и куки с Refresh токеном, если доступ разрешён.
    """
    # app.logger.info(f"XXX Incoming request: {request.method} {request.url}")
    # app.logger.debug(f"XXX Headers: {request.headers}")
    # app.logger.debug(f"XXX Body: {request.get_json()}")

    # Получние данных из тела запроса
    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')

    # Сравнение введённого пароля с паролем в бд
    user = db.actions.get_user(identifier)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({"msg": "Неверные данные"}), 401

    user_id = user['id']
    role = user['user_role']

    # Генерация JWT (токен для подтверждения, что пользователь уже авторизован)
    access_token = jwt.encode({
        "sub": str(user_id),
        "role": role,
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRES_MIN)).timestamp())
    }, SECRET_KEY, algorithm="HS256")

    # Генерация случайного Refresh токена (для продления доступа, т.е. перегенерации JWT)
    refresh_token = str(uuid.uuid4())
    expires_at_timestamp = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRES_DAYS)
    db.actions.insert_refresh_token(refresh_token, user_id, role, expires_at_timestamp)

    response = make_response(jsonify({"access_token": access_token, "role": role}))
    response.set_cookie("refresh_token", refresh_token, httponly=True, samesite='Strict', path="/")
    return response


@general_bp.route('/refresh', methods=['POST'])
def refresh():
    """
    Запрос на обновление JWT токена по Refresh токену.
    Запрос будет успешным, если Refresh токен есть в базе, его срок действия не истёк.\n
    Request: Refresh токен.\n
    Response: JWT токен.
    """
    # Проверка наличия Refresh токена в запросе
    refresh_token = request.cookies.get('refresh_token')
    if not refresh_token:
        return jsonify({"msg": "There is no refresh token"}), 401

    # Проверка наличия Refresh токена в базе
    record = db.actions.get_refresh_token(refresh_token)
    if not record:
        return jsonify({"msg": "Invalid refresh token"}), 401

    # Проверка срока годности Refresh токена
    expires_at = record['expires_at'].replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        db.actions.delete_refresh_token(refresh_token)
        return jsonify({"msg": "Refresh token expired"}), 401

    # Генерация нового токена
    access_token = jwt.encode({
        "sub": str(record['user_id']),
        "role": record['user_role'],
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRES_MIN)).timestamp())
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({"access_token": access_token})


@general_bp.route('/logout', methods=['POST'])
def logout():
    """
    Выход из системы. 
    Удаляется Refresh токен с сервера и куки с браузера.
    """
    refresh_token = request.cookies.get('refresh_token')
    if refresh_token:
        db.actions.delete_refresh_token(refresh_token)

    response = make_response(jsonify({"msg": "Logged out"}))
    response.delete_cookie("refresh_token")
    return response


@general_bp.route('/protected', methods=['GET'])
def protected():
    """Энд‑поинт для проверки JWT. Если JWT действителен, возвращает роль."""
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
