import mysql.connector
from pathlib import Path
from dotenv import load_dotenv
import os


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


def init_db():
    """Создание базы данных со всеми таблицами (создаёт только каркас без данных)."""
    execute_query(query=f"CREATE DATABASE IF NOT EXISTS {DB_NAME}", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD)
    execute_query(filename="schema.sql", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)


def clear_db():
    """Очистка базы данных (данные удаляются но каркас остаётся)."""
    execute_query(filename="clean.sql", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)


def seed_db():
    """Заполнение БД тестовыми данными (полностью перезаписывает все данные)."""
    clear_db()
    execute_query(filename="seed.sql", user=DB_ROOT_NAME, password=DB_ROOT_PASSWORD, database=DB_NAME)
