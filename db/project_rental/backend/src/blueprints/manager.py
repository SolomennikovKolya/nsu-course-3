from flask import Blueprint, request, jsonify
import db.actions

manager_bp = Blueprint('manager', __name__, url_prefix='/manager')


@manager_bp.route('/clients', methods=['GET'])
def get_clients():
    """Получение списка всех клиентов. Возвращаемые данные клиентов: id, name, phone, email."""
    result = db.actions.get_clients()
    return jsonify(result)


@manager_bp.route('/client_history', methods=['GET'])
def get_client_history():
    """Получение истории аренд клиента по ID клиента."""
    client_id = request.args.get('client_id')

    if not client_id:
        return jsonify({'error': 'client_id is required'}), 400

    result = db.actions.get_client_history(client_id)
    return jsonify(result)


@manager_bp.route('/all_equipment', methods=['GET'])
def get_all_equipment():
    """Получение списка всего оборудования."""
    result = db.actions.get_all_equipment()
    return jsonify(result)


@manager_bp.route('/items', methods=['GET'])
def get_all_items():
    """Получение списка айтемов выбранного обрудования."""
    equipment_id = request.args.get('equipment_id')
    result = db.actions.get_items(equipment_id)
    return jsonify(result)


@manager_bp.route('/add_item', methods=['POST'])
def add_item():
    """Добавление айтема."""
    equipment_id = request.json.get('equipment_id')
    db.actions.add_item(equipment_id)
    return jsonify({"msg": "Item added successfully"})


@manager_bp.route('/change_item_status', methods=['POST'])
def change_item_status():
    """Изменение статуса айтема."""
    item_id = request.json.get('item_id')
    status = request.json.get('status')
    db.actions.change_item_status(item_id, status)
    return jsonify({"msg": "Item added successfully"})


@manager_bp.route('/delete_item', methods=['POST'])
def delete_item():
    """Удаление айтема."""
    item_id = request.json.get('item_id')
    db.actions.delete_item(item_id)
    return jsonify({"msg": "Item deleted successfully"})
