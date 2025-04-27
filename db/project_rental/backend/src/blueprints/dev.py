from flask import Blueprint, jsonify
import db.actions


dev_bp = Blueprint('dev', __name__, url_prefix='/dev')


@dev_bp.route('/clear_db', methods=['POST'])
def clear_db():
    """Очистка базы данных."""
    try:
        db.actions.clear_db()
        return jsonify({"msg": "The database has been successfully cleaned"}), 200
    except Exception as e:
        return jsonify({"msg": f"Database error: {e}"}), 501


@dev_bp.route('/seed_db', methods=['POST'])
def seed_db():
    """Заполнение базы данных тестовыми данными."""
    try:
        db.actions.seed_db()
        return jsonify({"msg": "The database has been successfully filled in"}), 200
    except Exception as e:
        return jsonify({"msg": f"Database error: {e}"}), 501
