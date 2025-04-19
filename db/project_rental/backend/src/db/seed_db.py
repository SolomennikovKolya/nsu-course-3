import mysql.connector
from pathlib import Path
from flask import current_app


def seed_db():
    """Заполнение БД тестовыми данными."""
    DB_ROOT_NAME = current_app.config['DB_ROOT_NAME']
    DB_ROOT_PASSWORD = current_app.config['DB_ROOT_PASSWORD']
    DB_HOST = current_app.config['DB_HOST']
    DB_NAME = current_app.config['DB_NAME']

    seed_db_path = Path(__file__).resolve().parent / "seed_db.sql"
    with open(seed_db_path, "r", encoding="utf-8") as f:
        seed_sql = f.read()

    try:
        conn = mysql.connector.connect(
            user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME, autocommit=False)
        cursor = conn.cursor()
        conn.database = DB_NAME

        # MySQL автоматически начинает транзакцию, если использовать DML (INSERT, UPDATE, DELETE)
        # Изменения в БД происходят, но не фиксируются, пока транзакция открыта. Только conn.commit() фиксирует изменения
        conn.start_transaction()
        for statement in seed_sql.split(';'):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)
        conn.commit()

    except mysql.connector.Error as err:
        print(f"Error seeding database: {err}")

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
