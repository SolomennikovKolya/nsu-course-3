from flask import Blueprint, jsonify
import db.actions


dev_bp = Blueprint('dev', __name__, url_prefix='/dev')


@dev_bp.route('/init_db', methods=['POST'])
def init_db():
    """Создание базы данных."""
    db.actions.init_db()
    return jsonify({"msg": "The database has been successfully created"})


@dev_bp.route('/seed_db', methods=['POST'])
def seed_db():
    """Заполнение базы данных тестовыми данными."""
    db.actions.seed_db()
    return jsonify({"msg": "The database has been successfully filled in"})


@dev_bp.route('/clear_db', methods=['POST'])
def clear_db():
    """Очистка базы данных."""
    db.actions.clear_db()
    return jsonify({"msg": "The database has been successfully cleaned"})


@dev_bp.route('/drop_db', methods=['POST'])
def drop_db():
    """Очистка базы данных."""
    db.actions.drop_db()
    return jsonify({"msg": "The database has been successfully dropped"})
