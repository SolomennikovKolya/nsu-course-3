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
    const [modalError, setModalError] = useState(null);

    useEffect(() => {
        fetchClients();
        fetchRentals();
        fetchBookings();
    }, []);

    useEffect(() => {
        fetchRentals();
        fetchBookings();
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
            const res = await axios.get('/manager/rentals', { params: { client_id: selectedClient } });
            setRentals(res.data);
        } catch (err) {
            console.error('Ошибка при загрузке аренд', err);
        }
    };

    const fetchBookings = async () => {
        try {
            const res = await axios.get('/manager/bookings', { params: { client_id: selectedClient } });
            setBookings(res.data);
        } catch (err) {
            console.error('Ошибка при загрузке броней', err);
        }
    };

    const convertDate = (dateStr) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('ru-RU');
    };

    // Статус аренды
    const rentalStatus = (status, endDate) => {
        const currentDate = new Date();
        if (status === 'completed') {
            return <td style={{ backgroundColor: 'gray', color: 'white', fontWeight: 'bold', width: '10rem' }}>Завершено</td>
        } else if (status === 'active' && new Date(endDate) >= currentDate) {
            return <td style={{ backgroundColor: 'orangered', color: 'white', fontWeight: 'bold', width: '10rem' }}>Активно</td>
        } else {
            return <td style={{ backgroundColor: 'red', color: 'white', fontWeight: 'bold', width: '10rem' }}>Просрочено</td>
        }
    };

    // Функция для подсветки статуса брони
    const reservationStatus = (status) => {
        if (status === 'active') {
            return <td style={{ backgroundColor: 'orange', color: 'white', fontWeight: 'bold', width: '10rem' }}>Активно</td>
        } else if (status === 'cancelled') {
            return <td style={{ backgroundColor: 'gray', color: 'white', fontWeight: 'bold', width: '10rem' }}>Отменено</td>
        } else {
            return <td style={{ backgroundColor: 'gray', color: 'white', fontWeight: 'bold', width: '10rem' }}>Завершено</td>
        }
    };

    // Функция для открытия модального окна
    const openModal = (data, type) => {
        setModalData(data);
        setModalAction(type);
    };

    // Обработчик для закрытия модального окна
    const handleCloseModal = (e) => {
        if (e.target.classList.contains('modal')) {
            setModalAction(null);
            setModalError(null);
        }
    };

    // Выполнение действия в модальном окне
    const handleModalSubmit = async () => {
        try {
            if (modalAction === 'complete') {
                await axios.post('/manager/rentals/complete', { rental_id: modalData.id });
            } else if (modalAction === 'extend') {
                await axios.post('/manager/rentals/extend', { rental_id: modalData.id, extend_date: extendDate });
            } else if (modalAction === 'penalty') {
                await axios.post('/manager/rentals/penalty', { rental_id: modalData.id, penalty_amount: penalty });
            } else if (modalAction === 'cancel') {
                await axios.post('/manager/bookings/cancel', { booking_id: modalData.id });
            } else if (modalAction === 'book') {
                await axios.post('/manager/bookings/activate', { booking_id: modalData.id });
            }
            setModalAction(null);
            setModalData({});
            setModalError(null);
            fetchRentals();
            fetchBookings();
        } catch (error) {
            setModalError(error?.response?.data?.error);
            console.error('Ошибка при выполнении действия', error);
        }
    };

    return (
        <div style={{ padding: '2rem' }}>
            <h1 className="page-title">Аренды и брони</h1>

            {/* Селектор клиента */}
            <select className='text-select' onChange={(e) => setSelectedClient(e.target.value)} value={selectedClient}>
                <option value="">Все клиенты</option>
                {clients.map((client) => (
                    <option key={client.id} value={client.id}>{client.name}</option>
                ))}
            </select>

            {/* Таблица аренд и броней */}
            {(rentals.length === 0 && bookings.length === 0) ? (
                <p className='subtext' style={{ marginTop: '1rem' }}>Аренд и броней ещё не было</p>
            ) : (
                <table>
                    <thead>
                        <tr>
                            <th className='no-hover'>Тип</th>
                            <th className='no-hover'>Клиент</th>
                            <th className='no-hover'>Оборудование</th>
                            <th className='no-hover'>Айтем</th>
                            <th className='no-hover'>Начало</th>
                            <th className='no-hover'>Конец</th>
                            <th className='no-hover'>Залог</th>
                            <th className='no-hover'>Штраф</th>
                            <th className='no-hover'>Сумма</th>
                            <th className='no-hover'>Статус</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rentals.map((rental) => (
                            <tr className='tr-hover' key={rental.id} onClick={() => openModal(rental, 'complete')}>
                                <td>Аренда</td>
                                <td>{clients.find(client => client.id === rental.client_id)?.name ?? rental.client_id}</td>
                                <td>{rental.equipment_name}</td>
                                <td>{rental.item_id}</td>
                                <td>{convertDate(rental.start_date)}</td>
                                <td>{convertDate(rental.status === 'completed' ? rental.actual_return_date : rental.extended_end_date || rental.end_date)}</td>
                                <td>{rental.deposit_paid}</td>
                                <td>{rental.penalty_amount}</td>
                                <td>{rental.total_cost}</td>
                                {rentalStatus(rental.status, rental.extended_end_date || rental.end_date)}
                            </tr>
                        ))}
                        {bookings.map((booking) => (
                            <tr className='tr-hover' key={booking.id} onClick={() => openModal(booking, 'book')}>
                                <td>Бронь</td>
                                <td>{clients.find(client => client.id === booking.client_id)?.name ?? booking.client_id}</td>
                                <td>{booking.equipment_name}</td>
                                <td></td>
                                <td>{convertDate(booking.start_date)}</td>
                                <td>{convertDate(booking.end_date)}</td>
                                <td></td>
                                <td></td>
                                <td></td>
                                {reservationStatus(booking.status)}
                            </tr>
                        ))}
                    </tbody>
                </table>)}

            {/* Модальное окно */}
            {modalAction && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content">
                        <h2>Действие по {modalAction === 'complete' || modalAction === 'extend' || modalAction === 'penalty' ? 'аренде' : 'брони'}</h2>

                        {modalError !== null && (
                            <p className='error-message'>{modalError}</p>
                        )}

                        {/* Для аренды */}
                        {(modalAction === 'complete' || modalAction === 'extend' || modalAction === 'penalty') && (
                            <div>
                                <select className='text-select' onChange={(e) => { setModalAction(e.target.value); setModalError(null); }} value={modalAction}>
                                    <option value="complete">Завершить аренду</option>
                                    <option value="extend">Продлить аренду</option>
                                    <option value="penalty">Добавить штраф</option>
                                </select>
                            </div>
                        )}

                        {/* Для продления аренды */}
                        {modalAction === 'extend' && (
                            <div>
                                <label>Новая дата окончания аренды:</label>
                                <input className='text-input' placeholder='Дата продления' type="date" value={extendDate} onChange={(e) => setExtendDate(e.target.value)} />
                            </div>
                        )}

                        {/* Для добавления штрафа */}
                        {modalAction === 'penalty' && (
                            <div>
                                <input className='text-input' placeholder='Сумма штрафа' type="number" value={penalty} onChange={(e) => setPenalty(e.target.value)} />
                            </div>
                        )}

                        {/* Для брони */}
                        {(modalAction === 'book' || modalAction === 'cancel') && (
                            <div>
                                <select className='text-select' onChange={(e) => { setModalAction(e.target.value); setModalError(null); }} value={modalAction}>
                                    <option value="book">Оформить аренду</option>
                                    <option value="cancel">Отменить</option>
                                </select>
                            </div>
                        )}

                        {/* Кнопки */}
                        <button className='action-button' onClick={handleModalSubmit}>
                            {modalAction === 'complete' ? 'Завершить аренду' :
                                modalAction === 'extend' ? 'Продлить аренду' :
                                    modalAction === 'penalty' ? 'Начислить штраф' :
                                        modalAction === 'cancel' ? 'Отменить бронь' : 'Оформить аренду'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default RentalsAndBookings;
