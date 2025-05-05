from flask import Blueprint, jsonify
import db.actions


dev_bp = Blueprint('dev', __name__, url_prefix='/dev')


@dev_bp.route('/init_db', methods=['POST'])
def init_db():
    """Создание базы данных."""
    db.actions.init_db()
    return jsonify({"msg": "База данных успешно создана"})


@dev_bp.route('/seed_db', methods=['POST'])
def seed_db():
    """Заполнение базы данных тестовыми данными."""
    db.actions.seed_db()
    return jsonify({"msg": "База данных успешно заполнена"})


@dev_bp.route('/clear_db', methods=['POST'])
def clear_db():
    """Очистка базы данных."""
    db.actions.clear_db()
    return jsonify({"msg": "База данных успешно очищена"})


@dev_bp.route('/drop_db', methods=['POST'])
def drop_db():
    """Очистка базы данных."""
    db.actions.drop_db()
    return jsonify({"msg": "База данных успешно удалена"})


@dev_bp.route('/make_seed_db', methods=['POST'])
def make_seed_db():
    """Сохранение всех данных из БД."""
    db.actions.make_seed_db()
    return jsonify({"msg": "Данные из базы успешно сохранены"})
