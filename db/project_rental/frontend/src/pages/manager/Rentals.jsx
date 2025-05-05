import React, { useEffect, useState } from 'react';
import axios from '../../axios';

function RentalsAndBookings() {
    const [clients, setClients] = useState([]);
    const [selectedClient, setSelectedClient] = useState('');
    const [rentals, setRentals] = useState([]);
    const [bookings, setBookings] = useState([]);
    const [modalAction, setModalAction] = useState(null);
    const [modalData, setModalData] = useState({});
    const [penalty, setPenalty] = useState('');
    const [extendDate, setExtendDate] = useState('');

    useEffect(() => {
        fetchClients();
    }, []);

    useEffect(() => {
        if (selectedClient) {
            fetchRentals();
            fetchBookings();
        }
    }, [selectedClient]);

    const fetchClients = async () => {
        try {
            const res = await axios.get('/manager/clients');
            setClients(res.data);
        } catch (err) {
            console.error('Ошибка при загрузке клиентов', err);
        }
    };

    const fetchRentals = async () => {
        try {
            const res = await axios.get('/rentals', { params: { client_id: selectedClient } });
            setRentals(res.data);
        } catch (err) {
            console.error('Ошибка при загрузке аренд', err);
        }
    };

    const fetchBookings = async () => {
        try {
            const res = await axios.get('/bookings', { params: { client_id: selectedClient } });
            setBookings(res.data);
        } catch (err) {
            console.error('Ошибка при загрузке броней', err);
        }
    };

    // Функция для открытия модального окна
    const openModal = (data, type) => {
        setModalData(data);
        setModalAction(type);
    };

    // Выполнение действия в модальном окне
    const handleModalSubmit = async () => {
        try {
            if (modalAction === 'complete') {
                await axios.post('/rentals/complete', { rental_id: modalData.id });
            } else if (modalAction === 'extend') {
                await axios.post('/rentals/extend', { rental_id: modalData.id, extend_date: extendDate });
            } else if (modalAction === 'penalty') {
                await axios.post('/rentals/penalty', { rental_id: modalData.id, penalty_amount: penalty });
            } else if (modalAction === 'cancel') {
                await axios.post('/bookings/cancel', { booking_id: modalData.id });
            } else if (modalAction === 'book') {
                await axios.post('/bookings/book', { equipment_id: modalData.equipment_id, client_id: selectedClient });
            }
            setModalData({});
            setModalAction(null);
            fetchRentals();
            fetchBookings();
        } catch (err) {
            console.error('Ошибка при выполнении действия', err);
        }
    };

    // Функция для подсветки статуса аренды
    const rentalStatusStyle = (status, endDate) => {
        const currentDate = new Date();
        if (status === 'completed') {
            return { backgroundColor: 'green' };
        } else if (status === 'active' && new Date(endDate) >= currentDate) {
            return { backgroundColor: 'orange' };
        } else {
            return { backgroundColor: 'red' };
        }
    };

    return (
        <div style={{ padding: '2rem' }}>
            <h1>Аренды и брони</h1>

            {/* Селектор клиента */}
            <select onChange={(e) => setSelectedClient(e.target.value)} value={selectedClient}>
                <option value="">Выберите клиента</option>
                {clients.map((client) => (
                    <option key={client.id} value={client.id}>{client.name}</option>
                ))}
            </select>

            {/* Таблица аренды */}
            <h2>Аренды</h2>
            <table>
                <thead>
                    <tr>
                        <th>Клиент</th>
                        <th>Оборудование</th>
                        <th>ID айтема</th>
                        <th>Начало</th>
                        <th>Конец</th>
                        <th>Залог</th>
                        <th>Штраф</th>
                        <th>Статус</th>
                    </tr>
                </thead>
                <tbody>
                    {rentals.map((rental) => (
                        <tr key={rental.id} onClick={() => openModal(rental, 'complete')}>
                            <td>{rental.client_name}</td>
                            <td>{rental.equipment_name}</td>
                            <td>{rental.item_id}</td>
                            <td>{rental.start_date}</td>
                            <td>{rental.status === 'completed' ? rental.actual_return_date : rental.end_date}</td>
                            <td>{rental.deposit}</td>
                            <td>{rental.penalty}</td>
                            <td style={rentalStatusStyle(rental.status, rental.end_date)}>{rental.status}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {/* Таблица броней */}
            <h2>Бронь</h2>
            <table>
                <thead>
                    <tr>
                        <th>Клиент</th>
                        <th>Оборудование</th>
                        <th>Начало</th>
                        <th>Конец</th>
                        <th>Статус</th>
                    </tr>
                </thead>
                <tbody>
                    {bookings.map((booking) => (
                        <tr key={booking.id} onClick={() => openModal(booking, 'book')}>
                            <td>{booking.client_name}</td>
                            <td>{booking.equipment_name}</td>
                            <td>{booking.start_date}</td>
                            <td>{booking.end_date}</td>
                            <td>{booking.status}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {/* Модальное окно */}
            {modalAction && (
                <div className="modal">
                    <div className="modal-content">
                        <h2>Действие по {modalAction === 'complete' ? 'аренде' : 'брони'}</h2>
                        {modalAction === 'penalty' && (
                            <div>
                                <label>Сумма штрафа:</label>
                                <input
                                    type="number"
                                    value={penalty}
                                    onChange={(e) => setPenalty(e.target.value)}
                                />
                            </div>
                        )}
                        {modalAction === 'extend' && (
                            <div>
                                <label>Дата продления:</label>
                                <input
                                    type="date"
                                    value={extendDate}
                                    onChange={(e) => setExtendDate(e.target.value)}
                                />
                            </div>
                        )}
                        <button onClick={handleModalSubmit}>
                            {modalAction === 'complete' ? 'Завершить аренду' : modalAction === 'extend' ? 'Продлить аренду' : modalAction === 'penalty' ? 'Начислить штраф' : modalAction === 'cancel' ? 'Отменить бронь' : 'Забронировать'}
                        </button>
                        <button onClick={() => setModalAction(null)}>Закрыть</button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default RentalsAndBookings;
