from flask import Blueprint, jsonify
import db.actions


client_bp = Blueprint('client', __name__, url_prefix='/')


@client_bp.route('/catalog', methods=['GET'])
def get_catalog():
    """Получение списка всех существующих категорий оборудования."""
    result = db.actions.get_categories()
    return jsonify(result)
