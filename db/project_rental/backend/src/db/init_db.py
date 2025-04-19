import mysql.connector
from mysql.connector import errorcode
from pathlib import Path
from flask import current_app


def init_db():
    """Создание базы данных со всеми таблицами."""
    DB_ROOT_NAME = current_app.config['DB_ROOT_NAME']
    DB_ROOT_PASSWORD = current_app.config['DB_ROOT_PASSWORD']
    DB_HOST = current_app.config['DB_HOST']
    DB_NAME = current_app.config['DB_NAME']

    schema_path = Path(__file__).resolve().parent / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    try:
        # conn - соединение с БД (канал общения между приложением и сервером БД). Что делает conn:
        # Устанавливает TCP-соединение с MySQL; Управляет транзакциями (conn.commit(), conn.rollback()); Даёт доступ к курсору
        # cursor - объект для выполнения SQL-запросов внутри соединения
        # По умолчанию соединение с сервером MySQL начинается с включённого режима автоматической фиксации, который автоматически фиксирует каждый SQL-оператор
        # conn.database (или USE в sql) переключает контекст SQL-сессии на выбранную БД, к ней будут применяться все последующие команды
        conn = mysql.connector.connect(
            user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, autocommit=True)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.database = DB_NAME

        for statement in schema_sql.split(';'):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Access denied: check user/password.")
        else:
            print(f"Error: {err}")

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
