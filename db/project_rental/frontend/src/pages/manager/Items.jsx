import React, { useEffect, useState } from 'react';
import { useAuth } from '../../AuthContext';
import axios from '../../axios';

function Rentals() {
    const { user } = useAuth();
    const [equipment, setEquipment] = useState([]);                     // Список всего оборудования
    const [items, setItems] = useState([]);                             // Список айтемов для отображения в таблице
    const [selectedEquipmentID, setSelectedEquipmentID] = useState(""); // Выбранный фильтр оборудования
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalError, setModalError] = useState(null);
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
    const [newItem, setNewItem] = useState({ equipment_id: '' });

    // Загружаем данные при монтировании компонента
    useEffect(() => {
        if (user && (user.role === 'manager' || user.role === 'admin')) {
            fetchEquipments();
        }
    }, [user]);

    // Обновление списка всего оборудования
    const fetchEquipments = async () => {
        try {
            const res = await axios.get('/manager/all_equipment');
            setEquipment(res.data);
        } catch (err) {
            console.error('Ошибка загрузки оборудования:', err);
            alert('Не удалось загрузить оборудование');
        }
    };

    // Обновление списка айтемов
    const fetchItems = async () => {
        try {
            const res = await axios.get('/manager/items', {
                params: { equipment_id: selectedEquipmentID }
            });
            setItems(res.data);
        } catch (err) {
            console.error('Ошибка загрузки айтемов:', err);
            alert('Не удалось загрузить айтемы');
        }
    };

    // Получение списка айтемов выбранного оборудования
    useEffect(() => {
        fetchItems();
    }, [selectedEquipmentID]);

    // Добавление нового айтема
    const handleAddItem = async (e) => {
        e.preventDefault();
        try {
            await axios.post(`/manager/add_item`, { equipment_id: newItem.equipment_id });
            setIsModalOpen(false);
            setModalError(null);
            fetchItems();
        } catch (error) {
            console.error('Ошибка при добавлении айтема:', error);
            setModalError(error?.response?.data?.error);
        }
    };

    // Удаление айтема
    const handleDeleteItem = async (id) => {
        try {
            await axios.post(`/manager/delete_item`, { item_id: id });
            fetchItems();
        } catch (err) {
            console.error('Ошибка при удалении айтема:', err);
        }
    };

    // Обработчик для сортировки
    const requestSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const sortedItems = React.useMemo(() => {
        if (!sortConfig.key) return items;
        return [...items].sort((a, b) => {
            if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
            if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [items, sortConfig]);

    // Перевод статусов на русский
    const translateStatus = (status) => {
        switch (status) {
            case 'available': return 'Доступено';
            case 'booked': return 'Забронировано';
            case 'rented': return 'Арендовано';
            case 'serviced': return 'На обслуживании';
            case 'decommissioned': return 'Списано';
            default: return status;
        }
    };

    // Логика изменения статуса айтема
    const changeStatus = async (id, currentStatus) => {
        let newStatus;
        switch (currentStatus) {
            case 'available':
                newStatus = 'serviced';
                break;
            case 'serviced':
                newStatus = 'decommissioned';
                break;
            case 'decommissioned':
                newStatus = 'available';
                break;
            case 'booked':
                alert("Нельзя изменить статус забронированного оборудования");
                return;
            case 'rented':
                alert("Нельзя изменить статус арендованного оборудования");
                return;
            default:
                return;
        }

        try {
            await axios.post(`/manager/change_item_status`, { item_id: id, status: newStatus });
            fetchItems();
        } catch (err) {
            console.error('Ошибка при изменении статуса:', err);
        }
    };

    // Обработчик изменений в форме модального окна
    const handleChange = (e) => {
        const { name, value } = e.target;
        setNewItem((prev) => ({ ...prev, [name]: value }));
    };

    // Обработчик для закрытия модального окна
    const handleCloseModal = (e) => {
        if (e.target.classList.contains('modal')) {
            setIsModalOpen(false);
        }
    };

    if (!user || (user.role !== 'manager' && user.role !== 'admin')) {
        return <div style={{ padding: '2rem' }}><h1>Доступ запрещён</h1></div>;
    }

    return (
        <div style={{ padding: '2rem' }}>
            <h1 className="page-title">Единицы оборудования</h1>

            {/* Фильтр по оборудованию */}
            <select className='text-select' onChange={(e) => setSelectedEquipmentID(e.target.value)} value={selectedEquipmentID}>
                <option value="">Всё оборудование</option>
                {equipment.map((equipment) => (
                    <option key={equipment.id} value={equipment.id}>{equipment.name}</option>
                ))}
            </select>

            {/* Таблица айтемов */}
            <table>
                <thead>
                    <tr>
                        <th onClick={() => requestSort('id')}>ID {sortConfig.key === 'id' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th onClick={() => requestSort('equipment_name')}>Оборудование {sortConfig.key === 'equipment_name' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th onClick={() => requestSort('status')}>Статус {sortConfig.key === 'status' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th className='no-hover'></th>
                    </tr>
                </thead>
                <tbody>
                    {sortedItems.map((item) => (
                        <tr key={item.id}>
                            <td>{item.id}</td>
                            <td>{item.equipment_name}</td>
                            <td onClick={() => changeStatus(item.id, item.status)} className='changeable-cell'>
                                {translateStatus(item.status)}
                            </td>
                            <td style={{ padding: 0 }}>
                                <div onClick={() => handleDeleteItem(item.id)} className="delete-cell">X</div>
                            </td>
                        </tr>
                    ))}
                    <tr onClick={() => setIsModalOpen(true)} className="add-row">
                        <td colSpan="4" className="add-cell">Добавить айтем</td>
                    </tr>
                </tbody>
            </table>

            {/* Модальное окно для добавления айтема */}
            {isModalOpen && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content">
                        <h2>Добавить новый айтем</h2>
                        {modalError !== null && (<p className='error-message'>{modalError}</p>)}
                        <form onSubmit={handleAddItem}>
                            <select className='text-select' name="equipment_id" value={newItem.equipment_id} onChange={handleChange} required>
                                <option value="" disabled selected>Выберите оборудование</option>
                                {equipment.map((unit) => (
                                    <option key={unit.id} value={unit.id}>
                                        {unit.name}
                                    </option>
                                ))}
                            </select>
                            <button type="submit" className="action-button">Добавить</button>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Rentals;
