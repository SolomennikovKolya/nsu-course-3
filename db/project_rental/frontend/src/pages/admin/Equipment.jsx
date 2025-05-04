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
            setModalError(error?.response?.data?.error)
        }
    };

    const handleDeleteEquipment = async (id) => {
        try {
            await axios.post('/admin/equipment/delete', { id });
            fetchEquipment();
        } catch (error) {
            console.error('Ошибка при удалении оборудования:', error);
            alert(error?.response?.data?.error)
        }
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setNewEquipment((prev) => ({ ...prev, [name]: value }));
    };

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
                        <th>ID</th>
                        <th>Название</th>
                        <th>Категория</th>
                        <th>Описание</th>
                        <th>Цена аренды</th>
                        <th>Штраф за просрочку</th>
                        <th>Залог</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {equipment.map((unit) => (
                        <tr key={unit.id}>
                            <td>{unit.id}</td>
                            <td>{unit.name}</td>
                            <td>{unit.category}</td>
                            <td>{unit.description}</td>
                            <td>{unit.rental_price_per_day}</td>
                            <td>{unit.penalty_per_day}</td>
                            <td>{unit.deposit_amount}</td>
                            <td style={{ padding: 0 }}> <div onClick={() => handleDeleteEquipment(unit.id)} className="delete-cell">X</div></td>
                        </tr>
                    ))}
                    <tr onClick={() => setIsModalOpen(true)} className="add-row">
                        <td colSpan="8" className="add-cell">Добавить оборудование</td>
                    </tr>
                </tbody>
            </table>

            {/* Форма добавления сотрудника */}
            {isModalOpen && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h2>Добавить новое оборудование</h2>
                        {modalError !== null && (<p className='error-message'>{modalError}</p>)}
                        <form onSubmit={handleAddEquipment}>
                            <div>
                                <label>Характеристики оборудования:</label>
                                <input className='text-input' placeholder='Название' type="text" name="name" value={newEquipment.name} onChange={handleChange} required />
                                <input className='text-input' placeholder='Категория' type="tel" name="category" value={newEquipment.category} onChange={handleChange} required />
                                <input className='text-input' placeholder='Описание' type="description" name="description" value={newEquipment.description} onChange={handleChange} />
                                <input className='text-input' placeholder='Стоимость аренды' type="rental_price_per_day" name="rental_price_per_day" value={newEquipment.rental_price_per_day} onChange={handleChange} required />
                                <input className='text-input' placeholder='Штраф за просрочку' type="penalty_per_day" name="penalty_per_day" value={newEquipment.penalty_per_day} onChange={handleChange} required />
                                <input className='text-input' placeholder='Залог' type="deposit_amount" name="deposit_amount" value={newEquipment.deposit_amount} onChange={handleChange} required />
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
