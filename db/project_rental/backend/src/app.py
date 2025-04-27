from flask import Flask
from flask_cors import CORS
from cli.commands import register_commands
from blueprints.dev import dev_bp
from blueprints.general import general_bp
from blueprints.manager import manager_bp


# Настройка приложения
app = Flask(__name__)                 # app - экземпляр flask приложения
CORS(app, supports_credentials=True)  # CORS (Cross-Origin Resource Sharing) нужен, чтобы разрешить запросы с другого домена или порта
app.logger.setLevel("DEBUG")          # Уровени логирования: DEBUG, INFO, WARNING, ERROR, CRITICAL
register_commands(app)                # Регистрация Flask CLI команд
app.register_blueprint(dev_bp)        # Регистрирация блюпринта с dev командами
app.register_blueprint(general_bp)    # Основные запросы
app.register_blueprint(manager_bp)    # Запросы, реализующие обязанности мэнеджера


if __name__ == "__main__":
    app.run(debug=True)
