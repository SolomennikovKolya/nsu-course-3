from flask import Blueprint, request, jsonify
import db.actions


client_bp = Blueprint('client', __name__, url_prefix='/')


@client_bp.route('/catalog', methods=['GET'])
def get_catalog():
    """Получение списка всех существующих категорий оборудования."""
    result = db.actions.get_categories()
    return jsonify(result)


@client_bp.route('/category', methods=['GET'])
def get_category_equipment():
    """Получение списка оборудования по названию категории."""
    category_name = request.args.get('name')
    if not category_name:
        return jsonify({'error': 'Category name is required'}), 400

    result = db.actions.get_equipment_by_category(category_name)
    return jsonify(result)


@client_bp.route('/equipment', methods=['GET'])
def get_equipment():
    """Получение информации об оборудовании по его названию."""
    equipment_name = request.args.get('name')
    if not equipment_name:
        return jsonify({'error': 'Equipment name is required'}), 400

    result = db.actions.get_equipment_by_name(equipment_name)
    return jsonify(result)
