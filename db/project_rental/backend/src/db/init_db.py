import mysql.connector
from mysql.connector import errorcode
from mysql.connector.cursor import MySQLCursor
import os
from pathlib import Path


# Название базы данных (константа)
DB_NAME = "equipment_rental"


def init_db():
    """Инициализация базы данных. Если БД не было, она создаётся"""
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # schema_path = os.path.join(base_dir, "schema.sql")
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    with open(schema_path, "r") as f:
        SCHEMA_SQL = f.read()

    conn = None
    cursor = None
    try:
        # conn - соединение с БД (канал общения между приложением и сервером БД). Что делает conn:
        # Устанавливает TCP-соединение с MySQL; Управляет транзакциями (conn.commit(), conn.rollback()); Даёт доступ к курсору
        # cursor - объект для выполнения SQL-запросов внутри соединения
        # По умолчанию соединение с сервером MySQL начинается с включённого режима автоматической фиксации, который автоматически фиксирует каждый SQL-оператор
        # conn.database (USE в sql) переключает контекст SQL-сессии на выбранную БД, к ней будут применяться все последующие команды
        conn = mysql.connector.connect(
            user='root', password='root', host='localhost', autocommit=True)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.database = DB_NAME

        for statement in SCHEMA_SQL.split(';'):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)
        print("Database initialized and schema verified.")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Access denied: check user/password.")
        else:
            print(f"Error: {err}")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    init_db()
