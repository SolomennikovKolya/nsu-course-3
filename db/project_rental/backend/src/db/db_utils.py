from functools import wraps
import mysql.connector
from mysql.connector import errorcode


def handle_mysql_errors(log_func=print, re_raise=True):
    """
    Декоратор для обработки ошибок MySQL.
    Args:
        log_func (function): Функция логирования.
        re_raise (bool): Пробрасывать ли исключение дальше.
    Returns:
        None: ничего.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                    log_func(f"MySQL Error in {func.__name__}: \
                             Access denied: check user/password.")
                else:
                    log_func(f"MySQL Error in {func.__name__}: {err}")
                if re_raise:
                    raise
            except Exception as e:
                log_func(f"Unexpected error in {func.__name__}: {e}")
                if re_raise:
                    raise
        return wrapper
    return decorator
