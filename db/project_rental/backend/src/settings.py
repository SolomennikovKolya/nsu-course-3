import os
from dotenv import load_dotenv
from pathlib import Path
from flask import current_app


def load_config():
    """Загрузка настроек из .env в контекст приложения."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

    current_app.config['DB_NAME'] = os.getenv('DB_NAME')
    current_app.config['DB_HOST'] = os.getenv('DB_HOST', 'localhost')
    current_app.config['DB_ROOT_NAME'] = os.getenv('DB_ROOT_NAME')
    current_app.config['DB_ROOT_PASSWORD'] = os.getenv('DB_ROOT_PASSWORD')
