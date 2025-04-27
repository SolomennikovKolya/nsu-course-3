from flask import Blueprint, request, jsonify
import db.actions

manager_bp = Blueprint('manager', __name__, url_prefix='/manager')


@manager_bp.route('/clients', methods=['GET'])
def get_clients():
    """Получение списка всех клиентов."""
    result = db.actions.get_clients()
    return jsonify(result)


@manager_bp.route('/client_history', methods=['GET'])
def get_client_history():
    """Получение истории аренд клиента."""
    client_id = request.args.get('client_id')

    if not client_id:
        return jsonify({'error': 'client_id is required'}), 400

    result = db.actions.get_client_history(client_id)
    return jsonify(result)
