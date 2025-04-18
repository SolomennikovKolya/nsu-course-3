import mysql.connector

DB_NAME = "equipment_rental"


def seed_db():
    """Заполнение БД тестовыми данными"""
    with open("seed_data.sql", "r", encoding="utf-8") as f:
        seed_sql = f.read()

    try:
        conn = mysql.connector.connect(
            user='root', password='root', host='localhost', database=DB_NAME, autocommit=False)
        cursor = conn.cursor()
        conn.database = DB_NAME

        # MySQL автоматически начинает транзакцию, если использовать DML (INSERT, UPDATE, DELETE)
        # Изменения в БД происходят, но не фиксируются, пока транзакция открыта. Только conn.commit() фиксирует изменения
        conn.start_transaction()
        for result in cursor.execute(seed_sql, multi=True):
            if result.with_rows:
                print("Rows returned:", result.fetchall())
        conn.commit()
        print("Test data inserted.")

    except mysql.connector.Error as err:
        print(f"Error seeding database: {err}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    seed_db()
