from flask import Blueprint, request, jsonify
import db.actions

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/employee/get', methods=['GET'])
def get_clients():
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
    role = data.get('role')

    if not name:
        return jsonify({"error": "Имя - обязательное поле"}), 400
    if not phone:
        return jsonify({"error": "Телефон - обязательное поле"}), 400
    if not email:
        return jsonify({"error": "Email - обязательное поле"}), 400
    if not role:
        return jsonify({"error": "Роль - обязательное поле"}), 400

    db.actions.add_employee(name, phone, email, role)
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
