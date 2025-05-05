from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
import db.actions

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/employee/get', methods=['GET'])
def get_employee():
    """Получение списка всех сотрудников."""
    result = db.actions.get_employee()
    return jsonify(result)


@admin_bp.route('/employee/new', methods=['POST'])
def add_employee():
    """Добавление нового сотрудника."""
    data = request.get_json()

    name = data.get('name')
    phone = data.get('phone')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')

    if not name:
        return jsonify({"error": "Имя - обязательное поле"}), 400
    if not phone:
        return jsonify({"error": "Телефон - обязательное поле"}), 400
    if not email:
        return jsonify({"error": "Email - обязательное поле"}), 400
    if not role:
        return jsonify({"error": "Роль - обязательное поле"}), 400
    if not password:
        return jsonify({"error": "Пароль - обязательное поле"}), 400

    password_hash = generate_password_hash(password=password)
    if role == 'Менеджер':
        role = 'manager'
    elif role == 'Админ':
        role = 'admin'

    db.actions.add_employee(name, phone, email, password_hash, role)
    return jsonify({"msg": "Employee added successfully"})


@admin_bp.route('/employee/delete', methods=['POST'])
def delete_employee():
    """Удаление сотрудника по ID."""
    data = request.get_json()
    employee_id = data.get('id')

    if not employee_id:
        return jsonify({"error": "ID сотрудника - обязательное поле"}), 400

    db.actions.delete_employee(employee_id)
    return jsonify({"msg": "Employee deleted successfully"})


@admin_bp.route('/equipment/get', methods=['GET'])
def get_equipment():
    """Получение списка всего оборудования."""
    result = db.actions.get_equipment()
    return jsonify(result)


@admin_bp.route('/equipment/new', methods=['POST'])
def add_equipment():
    """Добавление нового оборудования."""
    data = request.get_json()

    name = data.get('name')
    category = data.get('category')
    description = data.get('description') or ""
    rental_price_per_day = data.get('rental_price_per_day')
    deposit_amount = data.get('deposit_amount')

    if not name:
        return jsonify({"error": "Название - обязательное поле"}), 400
    if not category:
        return jsonify({"error": "Категория - обязательное поле"}), 400
    if not rental_price_per_day:
        return jsonify({"error": "Цена аренды - обязательное поле"}), 400
    if not deposit_amount:
        return jsonify({"error": "Залог - обязательное поле"}), 400

    db.actions.add_equipment(name, category, description, rental_price_per_day, deposit_amount)
    return jsonify({"msg": "Equipment added successfully"})


@admin_bp.route('/equipment/delete', methods=['POST'])
def delete_equipment():
    """Удаление оборудования по ID."""
    data = request.get_json()
    equipment_id = data.get('id')

    if not equipment_id:
        return jsonify({"error": "ID оборудования - обязательное поле"}), 400

    result, code = db.actions.delete_equipment(equipment_id)
    return jsonify(result), code
