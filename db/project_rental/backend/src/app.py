from flask import Flask
from flask_cors import CORS
from auth import auth_bp
from flask import current_app
from settings import load_config
from db.init_db import init_db
from db.seed_db import seed_db


# app - экземпляр flask приложения
# CORS (Cross-Origin Resource Sharing) нужен, чтобы разрешить запросы с другого домена или порта
# "/api/*" ограничивает область действия CORS только на пути, начинающиеся с /api/
# "origins" указывает разрешённый источник — т.е. React-приложение, работающее на http://localhost:5173
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
with app.app_context():
    load_config()


# flask init_db - Инициализация БД
@app.cli.command("init_db")
def init_db_command():
    init_db()


# flask init_db - Заполнение БД тестовыми данными
@app.cli.command("seed_db")
def seed_db_command():
    seed_db()


# @app.route("/api/ping")
# def ping():
#     return {"message": "pong"}


if __name__ == "__main__":
    app.run(debug=True)
