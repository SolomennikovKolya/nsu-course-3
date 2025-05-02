import mysql.connector
from pathlib import Path
from dotenv import load_dotenv
import os
from functools import wraps


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


def get_queries_from_file(filename: str):
    """Получение всех запросов из файла в виде списка."""
    filename = Path(__file__).resolve().parent / "queries" / filename
    with open(filename, "r", encoding="utf-8") as f:
        queries_raw = f.read()

    queries = list()
    for statement in queries_raw.split(';'):
        stmt = statement.strip()
        if stmt:
            queries.append(stmt)
    return queries


def init_db():
    """Создание (либо полное пересоздание) базы данных со всеми таблицами (создаёт только каркас без данных)."""
    execute_query(query=f"DROP DATABASE IF EXISTS {DB_NAME}", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD)
    execute_query(query=f"CREATE DATABASE {DB_NAME}", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD)
    execute_query(filename="schema.sql", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)


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
        Rentals.penalty_amount,
        Rentals.status
    FROM Rentals
    JOIN Items ON Rentals.item_id = Items.id
    JOIN Equipment ON Items.equipment_id = Equipment.id
    WHERE Rentals.client_id = %s
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
            'penalty': record['penalty_amount'],
            'status': record['status'],
        })
    return result


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
