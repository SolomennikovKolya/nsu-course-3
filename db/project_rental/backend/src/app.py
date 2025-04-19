from flask import Flask
from flask_cors import CORS
import db.actions


# app - экземпляр flask приложения
# CORS (Cross-Origin Resource Sharing) нужен, чтобы разрешить запросы с другого домена или порта
# "/api/*" ограничивает область действия CORS только на пути, начинающиеся с /api/
# "origins" указывает разрешённый источник — т.е. React-приложение, работающее на http://localhost:5173
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})


# flask init_db
@app.cli.command("init_db")
def init_db():
    db.actions.init_db()


# flask clear_db
@app.cli.command("clear_db")
def clear_db():
    db.actions.clear_db()


# flask seed_db
@app.cli.command("seed_db")
def seed_db():
    db.actions.seed_db()


# @app.route("/api/ping")
# def ping():
#     return {"message": "pong"}


if __name__ == "__main__":
    app.run(debug=True)
