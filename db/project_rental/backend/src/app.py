from flask import Flask
from flask_cors import CORS
from auth import auth_bp

app = Flask(__name__)
# CORS (Cross-Origin Resource Sharing) нужен, чтобы разрешить запросы с другого домена или порта
# "/api/*" ограничивает область действия CORS только на пути, начинающиеся с /api/
# "origins": "http://localhost:5173" - указывает разрешённый источник — т.е. React-приложение, работающее на http://localhost:5173
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
app.register_blueprint(auth_bp, url_prefix='/api/auth')


@app.route("/api/ping")
def ping():
    return {"message": "pong"}


if __name__ == "__main__":

    # Пример использования
    # db_config = {
    #     'host': 'localhost',
    #     'user': 'root',
    #     'password': 'your_password',
    # }

    app.run(debug=True)
