import mysql.connector
from pathlib import Path
from dotenv import load_dotenv
import os
from functools import wraps
from datetime import datetime


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')
DB_ROOT_NAME = os.getenv('DB_ROOT_NAME')
DB_ROOT_PASSWORD = os.getenv('DB_ROOT_PASSWORD')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME')


def execute_query(query: str = None, filename: str = None, user: str = None, password: str = None,
                  host: str = DB_HOST, database: str = None, autocommit: bool = True):
    """
    Выполняет одну sql команду (указананную в query), либо все команды из файла (если query не указана, но указан filename).
    Args:
        query (str): Одна sql команда.
        filename (str): Название файла, который надо выполнить (например, "schema.sql").
        user (str): Пользователь, от имени которого происходит подключение к БД. У разных пользователей могут быть разные права.
        password (str): Пароль для подключения.
        host (str): Указывает, где находится сервер базы данных.
        database (str): Конкретная БД, в которой будут выполняться команды.
        autocommit (bool): Если True, то каждая sql команды будет как отдельная транзакция.
    """
    try:
        # conn - соединение с БД (канал общения между приложением и сервером БД). Что делает conn:
        # Устанавливает TCP-соединение с MySQL; Управляет транзакциями (conn.commit(), conn.rollback()); Даёт доступ к курсору
        # cursor - объект для выполнения SQL-запросов внутри соединения
        # По умолчанию соединение с сервером MySQL начинается с включённого режима автоматической фиксации, который автоматически фиксирует каждый SQL-оператор
        # conn.database (или USE в sql) переключает контекст SQL-сессии на выбранную БД, к ней будут применяться все последующие команды
        conn = mysql.connector.connect(user=user, password=password, host=host, database=database, autocommit=autocommit)
        cursor = conn.cursor()

        if query != None:
            cursor.execute(query)

        elif filename != None:
            filename = Path(__file__).resolve().parent / "queries" / filename
            with open(filename, "r", encoding="utf-8") as f:
                queries = f.read()
            for statement in queries.split(';'):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)

    except mysql.connector.Error as err:
        print(f"Query execution error: {err}")

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def with_db(user: str = None, password: str = None, host: str = None, database: str = None, autocommit: bool = True):
    """
    Декоратор, автоматизирующий работу с MySQL.
    Внутри функции, использующей данный декоратор, можно считать, что соединение уже создано и пользоваться conn, cursor для выполнения запросов.
    В параметрах декоратора указываются параметры подключения к бд.
    Подключение автоматически закрывается, незакоммиченные транзакции автоматически коммитятся.
    Если произошла ошибка, ролбэчит транзакции и также закрывает соединение, после чего пробрасывает исключение дальше.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn = cursor = None
            try:
                conn = mysql.connector.connect(
                    user=user,
                    password=password,
                    host=host,
                    database=database,
                    autocommit=autocommit
                )
                cursor = conn.cursor(dictionary=True)

                result = func(*args, conn=conn, cursor=cursor, **kwargs)
                if not autocommit:
                    conn.commit()
                return result

            except mysql.connector.Error as err:
                print(f"MySQL Error in {func.__name__}: {err}")
                if conn and not autocommit:
                    conn.rollback()
                raise

            except Exception as e:
                print(f"Unexpected error in {func.__name__}: {e}")
                if conn and not autocommit:
                    conn.rollback()
                raise

            finally:
                if cursor is not None:
                    cursor.close()
                if conn is not None:
                    conn.close()

        return wrapper
    return decorator


def get_queries_from_file(filename: str, delimiter: str = ';'):
    """Получение всех запросов из файла в виде списка."""
    filename = Path(__file__).resolve().parent / "queries" / filename
    with open(filename, "r", encoding="utf-8") as f:
        queries_raw = f.read()

    queries = list()
    for statement in queries_raw.split(delimiter):
        stmt = statement.strip()
        if stmt:
            queries.append(stmt)
    return queries


# +====================================================================================================+
# |------------------------------------------- DEVELOPMENT --------------------------------------------|
# +====================================================================================================+


def init_db():
    """Создание (либо полное пересоздание) базы данных со всеми таблицами (создаёт только каркас без данных)."""
    execute_query(query=f"DROP DATABASE IF EXISTS {DB_NAME}", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD)
    execute_query(query=f"CREATE DATABASE {DB_NAME}", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD)
    execute_query(filename="schema.sql", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)

    execute_query(query=f"DELIMITER //", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)
    for query in get_queries_from_file("triggers.sql", delimiter='//'):
        execute_query(query=query, user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)
    execute_query(query=f"DELIMITER ;", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME, autocommit=False)
def seed_db(conn, cursor):
    """Заполнение БД тестовыми данными (полностью перезаписывает все данные)."""
    for query in get_queries_from_file("clear.sql"):
        cursor.execute(query)
    for query in get_queries_from_file("seed.sql"):
        cursor.execute(query)


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def clear_db(conn, cursor):
    """Очистка базы данных (данные удаляются но каркас остаётся)."""
    for query in get_queries_from_file("clear.sql"):
        cursor.execute(query)


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def drop_db(conn, cursor):
    """Полное удаление базы данных."""
    cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def make_seed_db(conn, cursor):
    """Сохранение всех данных из БД в generated-seed.sql."""

    with open("./src/db/queries/generated-seed.sql", "w", encoding="utf-8") as f:
        # Запрос для извлечения данных из таблицы Equipment
        cursor.execute("SELECT * FROM Equipment")
        equipment_rows = cursor.fetchall()
        equipment_insert = "INSERT INTO Equipment (name, category, description, rental_price_per_day, deposit_amount) VALUES\n"

        # Формируем строки для вставки в таблицу Equipment
        equipment_values = []
        for row in equipment_rows:
            name = f"'{row['name']}'" if row['name'] is not None else 'NULL'
            category = f"'{row['category']}'" if row['category'] is not None else 'NULL'
            description = f"'{row['description']}'" if row['description'] is not None else 'NULL'
            rental_price_per_day = row['rental_price_per_day'] if row['rental_price_per_day'] is not None else 'NULL'
            deposit_amount = row['deposit_amount'] if row['deposit_amount'] is not None else 'NULL'

            values = f"({name}, {category}, {description}, {rental_price_per_day}, {deposit_amount})"
            equipment_values.append(values)
        if len(equipment_values) > 0:
            f.write(equipment_insert + ",\n".join(equipment_values) + ";\n\n")

        # Запрос для извлечения данных из таблицы Items
        cursor.execute("SELECT * FROM Items")
        items_rows = cursor.fetchall()
        items_insert = "INSERT INTO Items (equipment_id, status) VALUES\n"

        # Формируем строки для вставки в таблицу Items
        items_values = []
        for row in items_rows:
            equipment_id = row['equipment_id'] if row['equipment_id'] is not None else 'NULL'
            status = f"'{row['status']}'" if row['status'] is not None else 'NULL'
            values = f"({equipment_id}, {status})"
            items_values.append(values)
        if len(items_values) > 0:
            f.write(items_insert + ",\n".join(items_values) + ";\n\n")

        # Запрос для извлечения данных из таблицы Users
        cursor.execute("SELECT * FROM Users")
        users_rows = cursor.fetchall()
        users_insert = "INSERT INTO Users (user_role, password_hash, name, phone, email) VALUES\n"

        # Формируем строки для вставки в таблицу Users
        users_values = []
        for row in users_rows:
            user_role = f"'{row['user_role']}'" if row['user_role'] is not None else 'NULL'
            password_hash = f"'{row['password_hash']}'" if row['password_hash'] is not None else 'NULL'
            name = f"'{row['name']}'" if row['name'] is not None else 'NULL'
            phone = f"'{row['phone']}'" if row['phone'] is not None else 'NULL'
            email = f"'{row['email']}'" if row['email'] is not None else 'NULL'

            values = f"({user_role}, {password_hash}, {name}, {phone}, {email})"
            users_values.append(values)
        if len(users_values) > 0:
            f.write(users_insert + ",\n".join(users_values) + ";\n\n")

        # Запрос для извлечения данных из таблицы Reservations
        cursor.execute("SELECT * FROM Reservations")
        reservations_rows = cursor.fetchall()
        reservations_insert = "INSERT INTO Reservations (client_id, equipment_id, start_date, end_date, status) VALUES\n"

        # Формируем строки для вставки в таблицу Reservations
        reservations_values = []
        for row in reservations_rows:
            client_id = row['client_id'] if row['client_id'] is not None else 'NULL'
            equipment_id = row['equipment_id'] if row['equipment_id'] is not None else 'NULL'
            start_date = f"'{row['start_date']}'" if row['start_date'] is not None else 'NULL'
            end_date = f"'{row['end_date']}'" if row['end_date'] is not None else 'NULL'
            status = f"'{row['status']}'" if row['status'] is not None else 'NULL'

            values = f"({client_id}, {equipment_id}, {start_date}, {end_date}, {status})"
            reservations_values.append(values)
        if len(reservations_values) > 0:
            f.write(reservations_insert + ",\n".join(reservations_values) + ";\n\n")

        # Запрос для извлечения данных из таблицы Rentals
        cursor.execute("SELECT * FROM Rentals")
        rentals_rows = cursor.fetchall()
        rentals_insert = "INSERT INTO Rentals (client_id, item_id, start_date, end_date, extended_end_date, actual_return_date, deposit_paid, penalty_amount, total_cost, status) VALUES\n"

        # Формируем строки для вставки в таблицу Rentals
        rentals_values = []
        for row in rentals_rows:
            client_id = row['client_id'] if row['client_id'] is not None else 'NULL'
            item_id = row['item_id'] if row['item_id'] is not None else 'NULL'
            start_date = f"'{row['start_date']}'" if row['start_date'] is not None else 'NULL'
            end_date = f"'{row['end_date']}'" if row['end_date'] is not None else 'NULL'
            extended_end_date = f"'{row['extended_end_date']}'" if row['extended_end_date'] is not None else 'NULL'
            actual_return_date = f"'{row['actual_return_date']}'" if row['actual_return_date'] is not None else 'NULL'
            deposit_paid = row['deposit_paid'] if row['deposit_paid'] is not None else 'NULL'
            penalty_amount = row['penalty_amount'] if row['penalty_amount'] is not None else 'NULL'
            total_cost = row['total_cost'] if row['total_cost'] is not None else 'NULL'
            status = f"'{row['status']}'" if row['status'] is not None else 'NULL'

            values = f"({client_id}, {item_id}, {start_date}, {end_date}, {extended_end_date}, {actual_return_date}, {deposit_paid}, {penalty_amount}, {total_cost}, {status})"
            rentals_values.append(values)
        if len(rentals_values) > 0:
            f.write(rentals_insert + ",\n".join(rentals_values) + ";\n\n")

        # Запрос для извлечения данных из таблицы Refresh_Tokens
        cursor.execute("SELECT * FROM Refresh_Tokens")
        refresh_tokens_rows = cursor.fetchall()
        refresh_tokens_insert = "INSERT INTO Refresh_Tokens (token, user_id, user_role, expires_at) VALUES\n"

        # Формируем строки для вставки в таблицу Refresh_Tokens
        refresh_tokens_values = []
        for row in refresh_tokens_rows:
            token = f"'{row['token']}'" if row['token'] is not None else 'NULL'
            user_id = row['user_id'] if row['user_id'] is not None else 'NULL'
            user_role = f"'{row['user_role']}'" if row['user_role'] is not None else 'NULL'
            expires_at = f"'{row['expires_at']}'" if row['expires_at'] is not None else 'NULL'

            values = f"({token}, {user_id}, {user_role}, {expires_at})"
            refresh_tokens_values.append(values)
        if len(refresh_tokens_values) > 0:
            f.write(refresh_tokens_insert + ",\n".join(refresh_tokens_values) + ";\n\n")


# +====================================================================================================+
# |------------------------------------------- АВТОРИЗАЦИЯ --------------------------------------------|
# +====================================================================================================+


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_user(identifier, conn, cursor):
    """Получение пользователя из БД по имени, телефону или емейлу."""
    cursor.execute("SELECT * FROM Users WHERE name = %s OR phone = %s OR email = %s", (identifier, identifier, identifier))
    return cursor.fetchone()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def insert_refresh_token(token, user_id, user_role, expires_at, conn, cursor) -> str:
    """Добавление Refresh токена в БД."""
    cursor.execute(
        "INSERT INTO refresh_tokens(token, user_id, user_role, expires_at) "
        "VALUES (%s, %s, %s, %s)",
        (token, user_id, user_role, expires_at)
    )


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_refresh_token(token, conn, cursor):
    """Получение Refresh токена из БД."""
    cursor.execute(
        "SELECT user_id, user_role, expires_at "
        "FROM refresh_tokens WHERE token = %s",
        (token,)
    )
    return cursor.fetchone()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def delete_refresh_token(token, conn, cursor):
    """Удаление Refresh токена из БД."""
    cursor.execute("DELETE FROM refresh_tokens WHERE token = %s", (token,))


# +====================================================================================================+
# |---------------------------------------------- КЛИЕНТ ----------------------------------------------|
# +====================================================================================================+


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_categories(conn, cursor):
    """Возвращает список всех уникальных категорий оборудования."""
    cursor.execute("SELECT DISTINCT category FROM Equipment")
    categories = cursor.fetchall()
    return [{'name': category['category']} for category in categories]


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_equipment_by_category(category_name, conn, cursor):
    """Возвращает список оборудования для заданной категории."""
    cursor.execute("""
        SELECT name, rental_price_per_day
        FROM Equipment
        WHERE category = %s
    """, (category_name,))
    equipment = cursor.fetchall()
    return [{'name': item['name'], 'rental_price_per_day': item['rental_price_per_day']} for item in equipment]


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_equipment_by_name(equipment_name, conn, cursor):
    """Возвращает список оборудования для заданной категории с количеством доступных единиц."""
    cursor.execute("""
        SELECT e.name, e.rental_price_per_day, e.deposit_amount, e.description, COUNT(i.id) AS available_count
        FROM Equipment e LEFT JOIN Items i ON e.id = i.equipment_id AND i.status = 'available'
        WHERE e.name = %s
        GROUP BY e.id
    """, (equipment_name,))
    return cursor.fetchone()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME, autocommit=False)
def book_equipment(equipment_name, client_name, client_phone, client_email, start_date, end_date, conn, cursor):
    """Бронирование оборудования."""

    # Проверка, существует ли пользователь
    cursor.execute("""
        SELECT id FROM Users WHERE name = %s AND phone = %s AND email = %s
    """, (client_name, client_phone, client_email))
    user = cursor.fetchone()

    # Если пользователь не найден, добавляем нового
    user_id = None
    if not user:
        cursor.execute("""
            INSERT INTO Users (name, phone, email, user_role)
            VALUES (%s, %s, %s, 'client')
        """, (client_name, client_phone, client_email))
        user_id = cursor.lastrowid
    else:
        user_id = user['id']

    # Проверка доступности хотя бы одного элемента с данным оборудованием
    cursor.execute("""
        SELECT id FROM Equipment WHERE name = %s
    """, (equipment_name,))
    equipment = cursor.fetchone()
    if not equipment:
        return {"error": "Оборудование не найдено"}, 404

    equipment_id = equipment['id']
    cursor.execute("""
        SELECT id FROM Items WHERE equipment_id = %s AND status = 'available' LIMIT 1
    """, (equipment_id,))
    item = cursor.fetchone()
    if not item:
        return {"error": "Нет доступного оборудования для аренды"}, 400
    item_id = item['id']

    # Добавляем бронь в таблицу Reservations
    cursor.execute("""
        INSERT INTO Reservations (client_id, equipment_id, start_date, end_date, status)
        VALUES (%s, %s, %s, %s, 'active')
    """, (user_id, equipment_id, start_date, end_date))
    reservation_id = cursor.lastrowid

    return {"message": "Бронирование успешно"}, 200


# +====================================================================================================+
# |--------------------------------------------- МЕНЕДЖЕР ---------------------------------------------|
# +====================================================================================================+


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_rentals(client_id, conn, cursor):
    """Получение всех аренд или аренд определённого клиента (если client_id указан)."""
    cursor.execute("""
        SELECT r.id, r.client_id, r.item_id, r.start_date, r.end_date, r.extended_end_date, 
            r.actual_return_date, r.total_cost, r.deposit_paid, r.penalty_amount, r.status, i.equipment_id, 
            e.name AS equipment_name, e.category AS equipment_category
        FROM Rentals r
        JOIN Items i ON r.item_id = i.id
        JOIN Equipment e ON i.equipment_id = e.id
        WHERE (r.client_id = %s OR %s IS NULL)
        ORDER BY 
            FIELD(r.status, 'active', 'completed'),
            COALESCE(r.extended_end_date, r.end_date)
    """, (client_id, client_id))
    return cursor.fetchall()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def complete_rental(rental_id, conn, cursor):
    """Завершение аренды."""
    # Получаем информацию о аренде и оборудовании
    cursor.execute("""
        SELECT r.start_date, e.rental_price_per_day
        FROM Rentals r
        JOIN Items i ON r.item_id = i.id
        JOIN Equipment e ON i.equipment_id = e.id
        WHERE r.id = %s
    """, (rental_id,))

    rental = cursor.fetchone()
    if rental is None:
        return {"error": "Аренда не найдена"}, 404
    start_date = rental['start_date']
    rental_price_per_day = rental['rental_price_per_day']

    actual_return_date = datetime.now().date()
    total_cost = (actual_return_date - start_date).days * rental_price_per_day

    # Обновляем аренду
    cursor.execute("""
        UPDATE Rentals
        SET actual_return_date = %s, total_cost = %s, status = 'completed'
        WHERE id = %s
    """, (actual_return_date, total_cost, rental_id))

    return {"msg": "Аренда успешно завершена", "total_cost": total_cost}, 200


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def extend_rental(rental_id, extend_date, conn, cursor):
    """Продление аренды."""
    cursor.execute("SELECT end_date, extended_end_date FROM Rentals WHERE id = %s", (rental_id,))
    rental = cursor.fetchone()
    if rental is None:
        return {"error": "Аренда не найдена"}, 404

    extend_date = datetime.strptime(extend_date, '%Y-%m-%d').date()
    cur_end_date = rental['extended_end_date'] or rental['end_date']
    if (extend_date < cur_end_date):
        return {"error": "Дата продления должна быть больше текущей даты окончания аренды"}, 400

    cursor.execute("UPDATE Rentals SET extended_end_date = %s WHERE id = %s", (extend_date, rental_id))
    return {"msg": "Аренда успешно продлена"}, 200


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def penalty_rental(rental_id, penalty_amount, conn, cursor):
    """Начисление штрафа к аренде."""
    cursor.execute("""
        UPDATE Rentals 
        SET penalty_amount = penalty_amount + %s 
        WHERE id = %s
    """, (penalty_amount, rental_id))

    # Проверка, был ли обновлён штраф
    if cursor.rowcount == 0:
        return {"error": "Аренда не найдена"}, 404
    return {"msg": "Штраф успешно начислен"}, 200


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_bookings(client_id, conn, cursor):
    """Получение всех броней, либо броней определённого клиента (если указан client_id)."""
    cursor.execute("""
        SELECT r.id, r.client_id, r.equipment_id, r.start_date, r.end_date, r.status, 
            e.name AS equipment_name, e.category AS equipment_category
        FROM Reservations r
        JOIN Equipment e ON r.equipment_id = e.id
        WHERE (r.client_id = %s OR %s IS NULL)
        ORDER BY r.end_date
    """, (client_id, client_id))
    return cursor.fetchall()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def cancel_booking(booking_id, conn, cursor):
    """Отмена брони."""
    cursor.execute("SELECT status FROM Reservations WHERE id = %s", (booking_id,))
    data = cursor.fetchone()
    if data is None:
        return {"error": "Бронь не найдена"}, 404

    if data['status'] == 'completed':
        return {"error": "Бронь уже завершена"}, 400
    if data['status'] == 'cancelled':
        return {"error": "Бронь уже отменена"}, 400

    cursor.execute("UPDATE Reservations SET status = 'cancelled' WHERE id = %s", (booking_id, ))
    return {"msg": "Бронь успешно отменена"}, 200


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def activate_booking(booking_id, conn, cursor):
    """Переход от брони к аренде."""
    cursor.execute("SELECT status FROM Reservations WHERE id = %s", (booking_id,))
    data = cursor.fetchone()
    if data is None:
        return {"error": "Бронь не найдена"}, 404

    if data['status'] == 'completed':
        return {"error": "По завершённой брони нельзя оформить аренду"}, 400
    if data['status'] == 'cancelled':
        return {"error": "По отменённой брони нельзя оформить аренду"}, 400

    cursor.execute("UPDATE Reservations SET status = 'completed' WHERE id = %s", (booking_id, ))
    return {"msg": "Бронь успешно отменена"}, 200


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_items(equipment_id, conn, cursor):
    """Возвращает айтемов выбранного обрудования либо все айтемы."""
    if not equipment_id:
        cursor.execute("""SELECT i.id, e.name AS equipment_name, i.status 
            FROM Items i LEFT JOIN Equipment e ON i.equipment_id = e.id""")
    else:
        cursor.execute("""SELECT i.id, e.name AS equipment_name, i.status 
            FROM Items i LEFT JOIN Equipment e ON i.equipment_id = e.id WHERE i.equipment_id = %s""", (equipment_id,))
    return cursor.fetchall()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def add_item(equipment_id, conn, cursor):
    """Добавление айтема."""
    cursor.execute("""
        INSERT INTO Items (equipment_id, status)
        VALUES (%s, 'available')
    """, (equipment_id,))


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def change_item_status(item_id, status, conn, cursor):
    """Изменение статуса айтема."""
    cursor.execute("UPDATE Items SET status = %s WHERE id = %s", (status, item_id))


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def delete_item(item_id, conn, cursor):
    """Удаление айтема."""
    cursor.execute("SELECT status FROM Items WHERE id = %s", (item_id,))
    data = cursor.fetchone()
    if data is None:
        return {"error": "Айтем не найден"}, 404

    if data['status'] == 'rented':
        return {"error": "Нельзя удалить арендованный айтем"}, 400
    if data['status'] == 'booked':
        return {"error": "Нельзя удалить забронированный айтем"}, 400

    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("""DELETE FROM Items WHERE id = %s""", (item_id,))
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    return {"msg": "Айтем успешно удалён"}, 200


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_clients(conn, cursor):
    """Возвращает список всех клиентов."""
    cursor.execute("SELECT id, name, phone, email FROM Users WHERE user_role = 'client'")
    return cursor.fetchall()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_client_history(client_id, conn, cursor):
    """Возвращает историю аренд конкретного клиента."""
    query = """
    SELECT 
        Equipment.name AS equipment_name,
        Items.id AS item_id,
        Rentals.start_date,
        COALESCE(Rentals.extended_end_date, Rentals.end_date) AS end_date,
        Rentals.total_cost,
        Rentals.deposit_paid,
        Rentals.penalty_amount
    FROM Rentals
    JOIN Items ON Rentals.item_id = Items.id
    JOIN Equipment ON Items.equipment_id = Equipment.id
    WHERE Rentals.client_id = %s AND Rentals.status = 'completed'
    ORDER BY Rentals.start_date DESC
    """

    cursor.execute(query, (client_id,))
    history = cursor.fetchall()
    result = []
    for record in history:
        result.append({
            'equipment': record['equipment_name'],
            'item': record['item_id'],
            'start_date': record['start_date'].isoformat(),
            'end_date': record['end_date'].isoformat(),
            'rent_sum': record['total_cost'],
            'deposit': record['deposit_paid'],
            'penalty': record['penalty_amount']
        })
    return result


# +====================================================================================================+
# |---------------------------------------------- АДМИН -----------------------------------------------|
# +====================================================================================================+


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_equipment(conn, cursor):
    """Возвращает список всего оборудования."""
    cursor.execute("SELECT id, name, category, description, rental_price_per_day, deposit_amount FROM Equipment")
    return cursor.fetchall()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def add_equipment(name, category, description, rental_price_per_day, deposit_amount, conn, cursor):
    """Добавляет нового оборудования. Если название и категория новой записи совпадает со старой, то данные перезаписываются."""
    cursor.execute("""
        SELECT 1 FROM Equipment WHERE name = %s AND category = %s
    """, (name, category))

    if cursor.fetchone():
        cursor.execute("""
            UPDATE Equipment SET description = %s, rental_price_per_day = %s, deposit_amount = %s
            WHERE name = %s AND category = %s
        """, (description, rental_price_per_day, deposit_amount, name, category))
    else:
        cursor.execute("""
            INSERT INTO Equipment (name, category, description, rental_price_per_day, deposit_amount)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, category, description, rental_price_per_day, deposit_amount))


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME, autocommit=False)
def delete_equipment(equipment_id, conn, cursor):
    """Удаляет оборудование, если нет активных бронирований или аренды."""

    # Проверяем, есть ли активные бронирования для данного оборудования
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM Reservations
        WHERE equipment_id = %s AND status = 'active'
    """, (equipment_id,))
    active_reservations = cursor.fetchone()['cnt']
    if active_reservations > 0:
        return {"error": "Невозможно удалить оборудование, есть активные брони."}, 400

    # Проверяем, есть ли активные аренды для айтемов, связанных с данным оборудованием
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM Rentals r
        JOIN Items i ON r.item_id = i.id
        WHERE i.equipment_id = %s AND r.status = 'active'
    """, (equipment_id,))
    active_rentals = cursor.fetchone()['cnt']
    if active_rentals > 0:
        return {"error": "Невозможно удалить оборудование, есть активные аренды."}, 400

    # Если нет активных броней и аренды, удаляем оборудование и связанные записи
    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        cursor.execute("""DELETE FROM Rentals WHERE item_id IN (SELECT id FROM Items WHERE equipment_id = %s)""", (equipment_id,))
        cursor.execute("""DELETE FROM Reservations WHERE equipment_id = %s""", (equipment_id,))
        cursor.execute("""DELETE FROM Items WHERE equipment_id = %s""", (equipment_id,))
        cursor.execute("""DELETE FROM Equipment WHERE id = %s""", (equipment_id,))
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

        conn.commit()
        return {"message": "Оборудование и связанные записи удалены успешно."}, 200

    except Exception as e:
        conn.rollback()
        return {"error": f"Ошибка при удалении: {str(e)}"}, 500


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def get_employee(conn, cursor):
    """Возвращает список всех сотрудников."""
    cursor.execute("SELECT id, name, phone, email, user_role as role FROM Users WHERE user_role = 'manager' OR user_role = 'admin'")
    return cursor.fetchall()


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME)
def add_employee(name, phone, email, role, conn, cursor):
    """Добавляет нового сотрудника в таблицу Users. Если такой сотрудник уже есть, то ничего не меняется."""
    cursor.execute("""
        SELECT 1 FROM Users WHERE name = %s AND phone = %s AND email = %s AND user_role = %s
    """, (name, phone, email, role))
    if cursor.fetchone():
        return

    cursor.execute("""
        INSERT INTO Users (name, phone, email, user_role)
        VALUES (%s, %s, %s, %s)
    """, (name, phone, email, role))


@with_db(user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, host=DB_HOST, database=DB_NAME, autocommit=False)
def delete_employee(employee_id, conn, cursor):
    """Удаляет сотрудника по ID из таблицы Users."""
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("DELETE FROM Users WHERE id = %s", (employee_id,))
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
