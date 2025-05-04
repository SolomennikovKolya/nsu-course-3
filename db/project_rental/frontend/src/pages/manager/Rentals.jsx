import React, { useEffect, useState } from 'react';
import { useAuth } from '../../AuthContext';
import axios from '../../axios';
import styles from './Rentals.module.css';

function Rentals() {
    const { user } = useAuth();
    const [equipment, setEquipment] = useState([]);                     // Список всего оборудования
    const [clients, setClients] = useState([]);                         // Список всех клиентов
    const [items, setItems] = useState([]);                             // Список айтемов для отображения в таблице
    const [selectedEquipmentID, setSelectedEquipmentID] = useState(""); // Выбранный фильтр оборудования
    const [currentDate, setCurrentDate] = useState(new Date());
    const [selectedItem, setSelectedItem] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalError, setModalError] = useState(null);

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
    }

    // Получение списка айтемов выбранного обрудования
    useEffect(() => {
        fetchItems();
    }, [selectedEquipmentID]);

    // Добавление нового айтема
    const handleAddItem = async (e) => {
        e.preventDefault();

        try {
            // await axios.post('/admin/employee/new', newEmployee);
            alert("Айтем добавлен");
            setIsModalOpen(false);
            setModalError(null);
            fetchItems();
        } catch (error) {
            console.error('Ошибка при добавлении сотрудника:', error);
            setModalError(error?.response?.data?.error)
        }
    };

    const handleDeleteItem = async (id) => {
        try {
            // await axios.post('/admin/employee/delete', { id });
            alert("Айтем удалён");
            fetchItems();
        } catch (err) {
            console.error('Ошибка при удалении сотрудника:', err);
        }
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
        <div style={{ padding: '2rem', display: 'flex', gap: '2rem' }}>
            <div style={{ flex: 1 }}>
                {/* Фильтр по оборудованию */}
                <select className='text-select' onChange={(e) => setSelectedEquipmentID(e.target.value)} value={selectedEquipmentID}>
                    <option value="">Выберите оборудование</option>
                    {equipment.map((equipment) => (
                        <option key={equipment.id} value={equipment.id}>{equipment.name}</option>
                    ))}
                </select>

                {/* Таблица айтемов */}
                <table className='custom-table'>
                    <thead>
                        <tr>
                            <th>id</th>
                            <th>equipment_id</th>
                            <th>status</th>
                            <th>last_maintenance_date</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        {items.map((item) => (
                            <tr key={item.id}>
                                <td>{item.id}</td>
                                <td>{item.equipment_id}</td>
                                <td>{item.status}</td>
                                <td>{item.last_maintenance_date}</td>
                                <td style={{ padding: 0 }}> <div onClick={() => handleDeleteItem(item.id)} className={styles["delete-cell"]}>X</div></td>
                            </tr>
                        ))}
                        <tr onClick={() => setIsModalOpen(true)} className={styles["add-row"]}>
                            <td colSpan="5" className={styles["add-cell"]}>Добавить айтем</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            {/* Правая часть с выбранным айтемом */}
            <div style={{ width: '300px', flexShrink: 0 }}>
                {selectedItem && (
                    <div>
                        <h2>{selectedItem.equipment_name} : {selectedItem.id}</h2>
                        <p>Текущий статус: {selectedItem.status}</p>
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <button disabled={selectedItem.status === 'rented'}>Выдать</button>
                            <button disabled={selectedItem.status === 'available'}>Вернуть</button>
                        </div>
                    </div>
                )}
            </div>

            {/* Модальное окно для добавления айтема */}
            {isModalOpen && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content">
                        <h2>Добавить новый айтем</h2>
                        {modalError !== null && (<p className='error-message'>{modalError}</p>)}
                        <form onSubmit={handleAddItem}>
                            <select className='text-select'>
                                <option value="">Выберите оборудование</option>
                                {equipment.map((equipment) => (
                                    <option key={equipment.id} value={equipment.id}>
                                        {equipment.name}
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
