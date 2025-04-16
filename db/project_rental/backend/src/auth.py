from flask import Blueprint, request, jsonify

auth_bp = Blueprint('auth', __name__)

# Мок-данные пользователей
users = {
    "user": {"password": "123", "role": "user"},
    "manager": {"password": "123", "role": "manager"},
    "admin": {"password": "123", "role": "admin"},
}


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = users.get(username)
    if user and user["password"] == password:
        return jsonify({"username": username, "role": user["role"]}), 200
    return jsonify({"error": "Invalid credentials"}), 401
