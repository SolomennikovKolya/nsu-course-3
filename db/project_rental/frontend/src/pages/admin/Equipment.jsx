import React, { useEffect, useState } from 'react';
import { useAuth } from '../../AuthContext';
import axios from '../../axios';

function Employees() {
    const { user } = useAuth();
    const [equipment, setEquipment] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalError, setModalError] = useState(null);
    const [newEquipment, setNewEquipment] = useState({
        name: '', category: '', description: '',
        rental_price_per_day: '', penalty_per_day: '', deposit_amount: ''
    });
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

    useEffect(() => {
        if (user && user.role === 'admin') {
            fetchEquipment();
        }
    }, [user]);

    const fetchEquipment = async () => {
        try {
            const res = await axios.get('/admin/equipment/get');
            setEquipment(res.data);
        } catch (err) {
            console.error('Ошибка загрузки списка оборудования:', err);
        }
    };

    const handleAddEquipment = async (e) => {
        e.preventDefault();
        try {
            await axios.post('/admin/equipment/new', newEquipment);
            setIsModalOpen(false);
            setModalError(null);
            fetchEquipment();
        } catch (error) {
            console.error('Ошибка при добавлении оборудования:', error);
            setModalError(error?.response?.data?.error);
        }
    };

    const handleDeleteEquipment = async (id) => {
        try {
            await axios.post('/admin/equipment/delete', { id });
            fetchEquipment();
        } catch (error) {
            console.error('Ошибка при удалении оборудования:', error);
            alert(error?.response?.data?.error);
        }
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setNewEquipment((prev) => ({ ...prev, [name]: value }));
    };

    // Обработчик сортировки
    const requestSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const sortedEquipment = React.useMemo(() => {
        if (!sortConfig.key) return equipment;
        return [...equipment].sort((a, b) => {
            if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
            if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [equipment, sortConfig]);

    // Обработчик клика по фону модального окна (для его закрытия)
    const handleCloseModal = (e) => {
        if (e.target.classList.contains('modal')) {
            setIsModalOpen(false);
        }
    };

    if (!user || user.role !== 'admin') {
        return <div style={{ padding: '2rem' }}><h1>Доступ запрещён</h1></div>;
    }

    return (
        <div style={{ padding: '2rem' }}>
            <h1 className="page-title">Оборудование</h1>

            {/* Таблица оборудования */}
            <table>
                <thead>
                    <tr>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('id')}>ID {sortConfig.key === 'id' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('name')}>Название {sortConfig.key === 'name' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('category')}>Категория {sortConfig.key === 'category' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('description')}>Описание {sortConfig.key === 'description' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('rental_price_per_day')}>Цена аренды {sortConfig.key === 'rental_price_per_day' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('penalty_per_day')}>Штраф за просрочку {sortConfig.key === 'penalty_per_day' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th style={{ cursor: 'pointer' }} onClick={() => requestSort('deposit_amount')}>Залог {sortConfig.key === 'deposit_amount' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {sortedEquipment.map((unit) => (
                        <tr key={unit.id}>
                            <td>{unit.id}</td>
                            <td>{unit.name}</td>
                            <td>{unit.category}</td>
                            <td>{unit.description}</td>
                            <td>{unit.rental_price_per_day}</td>
                            <td>{unit.penalty_per_day}</td>
                            <td>{unit.deposit_amount}</td>
                            <td style={{ padding: 0 }}>
                                <div onClick={() => handleDeleteEquipment(unit.id)} className="delete-cell">X</div>
                            </td>
                        </tr>
                    ))}
                    <tr onClick={() => setIsModalOpen(true)} className="add-row">
                        <td colSpan="8" className="add-cell">Добавить оборудование</td>
                    </tr>
                </tbody>
            </table>

            {/* Форма добавления оборудования */}
            {isModalOpen && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h2>Добавить новое оборудование</h2>
                        {modalError !== null && (<p className='error-message'>{modalError}</p>)}
                        <form onSubmit={handleAddEquipment}>
                            <div>
                                <label>Характеристики оборудования:</label>
                                <input className='text-input' placeholder='Название' type="text" name="name" value={newEquipment.name} onChange={handleChange} required />
                                <input className='text-input' placeholder='Категория' type="text" name="category" value={newEquipment.category} onChange={handleChange} required />
                                <input className='text-input' placeholder='Описание' type="text" name="description" value={newEquipment.description} onChange={handleChange} />
                                <input className='text-input' placeholder='Стоимость аренды' type="number" name="rental_price_per_day" value={newEquipment.rental_price_per_day} onChange={handleChange} required />
                                <input className='text-input' placeholder='Штраф за просрочку' type="number" name="penalty_per_day" value={newEquipment.penalty_per_day} onChange={handleChange} required />
                                <input className='text-input' placeholder='Залог' type="number" name="deposit_amount" value={newEquipment.deposit_amount} onChange={handleChange} required />
                            </div>
                            <button type="submit" className="action-button">Добавить</button>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Employees;
