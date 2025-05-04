import React, { useEffect, useState } from 'react';
import { useAuth } from '../../AuthContext';
import axios from '../../axios';

function Employees() {
    const { user } = useAuth();
    const [employees, setEmployees] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalError, setModalError] = useState(null);
    const [newEmployee, setNewEmployee] = useState({ name: '', phone: '', email: '', role: '' });
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

    useEffect(() => {
        if (user && user.role === 'admin') {
            fetchEmployees();
        }
    }, [user]);

    const fetchEmployees = async () => {
        try {
            const res = await axios.get('/admin/employee/get');
            setEmployees(res.data);
        } catch (err) {
            console.error('Ошибка загрузки сотрудников:', err);
        }
    };

    const handleAddEmployee = async (e) => {
        e.preventDefault();
        try {
            await axios.post('/admin/employee/new', newEmployee);
            setIsModalOpen(false);
            setModalError(null);
            fetchEmployees();
        } catch (error) {
            console.error('Ошибка при добавлении сотрудника:', error);
            setModalError(error?.response?.data?.error)
        }
    };

    const handleDeleteEmployee = async (id) => {
        try {
            await axios.post('/admin/employee/delete', { id });
            fetchEmployees();
        } catch (err) {
            console.error('Ошибка при удалении сотрудника:', err);
        }
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setNewEmployee((prev) => ({ ...prev, [name]: value }));
    };

    // Обработчик сортировки
    const requestSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const sortedEmployees = React.useMemo(() => {
        if (!sortConfig.key) return employees;
        return [...employees].sort((a, b) => {
            if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
            if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [employees, sortConfig]);

    const translateRole = (role) => {
        switch (role) {
            case 'client': return 'Клиент';
            case 'manager': return 'Менеджер';
            case 'admin': return 'Админ';
            default: return role;
        }
    };

    // Обработчик для закрытия модального окна
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
            <h1 className="page-title">Сотрудники</h1>

            {/* Таблица сотрудников */}
            <table className='custom-table'>
                <thead>
                    <tr>
                        <th onClick={() => requestSort('id')}>ID {sortConfig.key === 'id' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th onClick={() => requestSort('name')}>Имя {sortConfig.key === 'name' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th onClick={() => requestSort('phone')}>Номер {sortConfig.key === 'phone' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th onClick={() => requestSort('email')}>Email {sortConfig.key === 'email' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th onClick={() => requestSort('role')}>Должность {sortConfig.key === 'role' ? (sortConfig.direction === 'asc' ? '▲' : '▼') : ''}</th>
                        <th className='no-hover'></th>
                    </tr>
                </thead>
                <tbody>
                    {sortedEmployees.map((employee) => (
                        <tr key={employee.id}>
                            <td>{employee.id}</td>
                            <td>{employee.name}</td>
                            <td>{employee.phone}</td>
                            <td>{employee.email}</td>
                            <td>{translateRole(employee.role)}</td>
                            <td style={{ padding: 0 }}>
                                <div onClick={() => handleDeleteEmployee(employee.id)} className="delete-cell">X</div>
                            </td>
                        </tr>
                    ))}
                    <tr onClick={() => setIsModalOpen(true)} className="add-row">
                        <td colSpan="6" className="add-cell">Добавить сотрудника</td>
                    </tr>
                </tbody>
            </table>

            {/* Форма добавления сотрудника */}
            {isModalOpen && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h2>Добавить нового сотрудника</h2>
                        {modalError !== null && (<p className='error-message'>{modalError}</p>)}
                        <form onSubmit={handleAddEmployee}>
                            <div>
                                <label>Данные сотрудника:</label>
                                <input className='text-input' placeholder='Имя' type="text" name="name" value={newEmployee.name} onChange={handleChange} required />
                                <input className='text-input' placeholder='Телефон' type="tel" name="phone" value={newEmployee.phone} onChange={handleChange} required />
                                <input className='text-input' placeholder='Email' type="email" name="email" value={newEmployee.email} onChange={handleChange} required />
                                <select className='text-select' name="role" value={newEmployee.role} onChange={handleChange} required>
                                    <option value="" disabled selected>Должность</option>
                                    <option value="manager">Менеджер</option>
                                    <option value="admin">Админ</option>
                                </select>
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
