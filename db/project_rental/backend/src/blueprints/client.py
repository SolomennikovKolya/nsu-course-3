from flask import Blueprint, request, jsonify
from datetime import datetime
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


@client_bp.route('/book_equipment', methods=['POST'])
def book_equipment():
    """Бронирование оборудования."""
    equipment_name = request.json.get('equipmentName')
    client_name = request.json.get('name')
    client_phone = request.json.get('phone')
    client_email = request.json.get('email')
    start_date = request.json.get('rentFrom')
    end_date = request.json.get('rentTo')

    if not equipment_name:
        return jsonify({'error': 'Необходимо указать название оборудования'}), 400
    if not client_name:
        return jsonify({'error': 'Имя - обязательное поле'}), 400
    if not client_phone:
        return jsonify({'error': 'Телефон - обязательное поле'}), 400
    if not client_email:
        return jsonify({'error': 'Email - обязательное поле'}), 400
    if not start_date:
        return jsonify({'error': 'Дата начала аренды - обязательное поле'}), 400
    if not end_date:
        return jsonify({'error': 'Дата окончания аренды - обязательное поле'}), 400

    try:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Неверный формат даты. Используйте формат YYYY-MM-DD.'}), 400

    now = datetime.now()
    # if start_date_obj < now:
    #     return jsonify({'error': 'Дата начала аренды не может быть в прошлом'}), 400
    if end_date_obj < start_date_obj:
        return jsonify({'error': 'Дата окончания аренды не может быть раньше даты начала аренды'}), 400

    result, code = db.actions.book_equipment(equipment_name, client_name, client_phone, client_email, start_date, end_date)
    return jsonify(result), code
