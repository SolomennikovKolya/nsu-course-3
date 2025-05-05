from flask import Blueprint, request, jsonify
import db.actions

manager_bp = Blueprint('manager', __name__, url_prefix='/manager')


@manager_bp.route('/rentals', methods=['GET'])
def get_rentals():
    """Получение всех аренд, либо аренд определённого клиента (если указан client_id)."""
    client_id = request.args.get('client_id') or None
    result = db.actions.get_rentals(client_id)
    return jsonify(result)


@manager_bp.route('/rentals/complete', methods=['POST'])
def complete_rental():
    # """Завершение аренды."""
    rental_id = request.json.get('rental_id')
    if not rental_id:
        return jsonify({'error': 'Необходимо указать rental_id'}), 400

    result, code = db.actions.complete_rental(rental_id)
    return jsonify(result), code


@manager_bp.route('/rentals/extend', methods=['POST'])
def extend_rental():
    # """Продление аренды."""
    rental_id = request.json.get('rental_id')
    if not rental_id:
        return jsonify({'error': 'Необходимо указать rental_id'}), 400

    db.actions.extend_rental(rental_id)
    return jsonify({"msg": "Аренда успешно завершена"})


@manager_bp.route('/rentals/penalty', methods=['POST'])
def penalty_rental():
    # """Начисление штрафа при аренде."""
    rental_id = request.json.get('rental_id')
    penalty_amount = request.json.get('penalty_amount')
    if not rental_id:
        return jsonify({'error': 'Необходимо указать rental_id'}), 400
    if not rental_id:
        return jsonify({'error': 'Необходимо указать penalty_amount'}), 400

    db.actions.penalty_rental(rental_id, penalty_amount)
    return jsonify({"msg": "Штраф успешно начислен"})


@manager_bp.route('/bookings', methods=['GET'])
def get_bookings():
    """Получение всех броней, либо броней определённого клиента (если указан client_id)."""
    client_id = request.args.get('client_id') or None
    result = db.actions.get_bookings(client_id)
    return jsonify(result)


@manager_bp.route('/bookings/cancel', methods=['POST'])
def cancel_booking():
    # """Отмена брони."""
    booking_id = request.json.get('booking_id')
    if not booking_id:
        return jsonify({'error': 'Необходимо указать booking_id'}), 400

    db.actions.cancel_booking(booking_id)
    return jsonify({"msg": "Бронь успешно отменена"})


@manager_bp.route('/bookings/activate', methods=['POST'])
def activate_booking():
    # """Переход от брони к аренде."""
    booking_id = request.json.get('booking_id')
    if not booking_id:
        return jsonify({'error': 'Необходимо указать booking_id'}), 400

    db.actions.activate_booking(booking_id)
    return jsonify({"msg": "Бронь успешно стала арендой"})


@manager_bp.route('/all_equipment', methods=['GET'])
def get_all_equipment():
    """Получение списка всего оборудования."""
    result = db.actions.get_equipment()
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
