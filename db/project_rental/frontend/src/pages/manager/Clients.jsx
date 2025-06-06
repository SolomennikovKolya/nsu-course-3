import React, { useEffect, useState } from 'react';
import { useAuth } from '../../AuthContext';
import axios from '../../axios';
import styles from './Clients.module.css';

function Clients() {
    const { user } = useAuth();
    const [clients, setClients] = useState([]);
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
    const [selectedClient, setSelectedClient] = useState(null);
    const [clientHistory, setClientHistory] = useState([]);
    const [loadingClients, setLoadingClients] = useState(false);
    const [loadingHistory, setLoadingHistory] = useState(false);

    useEffect(() => {
        if (user && (user.role === 'manager' || user.role === 'admin')) {
            fetchClients();
        }
    }, [user]);

    const fetchClients = async () => {
        try {
            setLoadingClients(true);
            const res = await axios.get('/manager/clients');
            setClients(res.data);
        } catch (err) {
            console.error('Ошибка загрузки клиентов:', err);
        } finally {
            setLoadingClients(false);
        }
    };

    const fetchClientHistory = async (clientId) => {
        try {
            setLoadingHistory(true);
            const res = await axios.get('/manager/client_history', { params: { client_id: clientId } });
            setClientHistory(res.data);
        } catch (err) {
            console.error('Ошибка загрузки истории клиента:', err);
        } finally {
            setLoadingHistory(false);
        }
    };

    const requestSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const sortedClients = React.useMemo(() => {
        if (!sortConfig.key) return clients;
        return [...clients].sort((a, b) => {
            if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
            if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [clients, sortConfig]);

    const handleRowClick = async (client) => {
        setSelectedClient(client);
        await fetchClientHistory(client.id);
    };

    const closeModal = () => {
        setSelectedClient(null);
        setClientHistory([]);
    };

    const getSortArrow = (key) => {
        if (sortConfig.key !== key) return '';
        return sortConfig.direction === 'asc' ? '▲' : '▼';
    };

    if (!user || (user.role !== 'manager' && user.role !== 'admin')) {
        return <div style={{ padding: '2rem' }}><h1>Доступ запрещён</h1></div>;
    }

    return (
        <div style={{ padding: '2rem' }}>
            <h1 className='page-title'>Клиенты</h1>

            {/* Таблица клиентов */}
            {loadingClients ? (
                <div></div>
            ) : (
                <table className={styles['clients-table']}>
                    <thead>
                        <tr>
                            <th onClick={() => requestSort('id')}>ID {getSortArrow('id')}</th>
                            <th onClick={() => requestSort('name')}>Имя {getSortArrow('name')}</th>
                            <th onClick={() => requestSort('phone')}>Номер {getSortArrow('phone')}</th>
                            <th onClick={() => requestSort('email')}>Email {getSortArrow('email')}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedClients.map((client) => (
                            <tr key={client.id} onClick={() => handleRowClick(client)} className={styles['clients-table-tr-hover']}>
                                <td>{client.id}</td>
                                <td>{client.name}</td>
                                <td>{client.phone}</td>
                                <td>{client.email}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}

            {/* Модальное окно истории аренд */}
            {selectedClient && (
                <div className={styles['history-modal']} onClick={closeModal}>
                    <div className={styles['history-modal-content']} onClick={(e) => e.stopPropagation()}>
                        <h2>История аренд - {selectedClient.name}</h2>

                        {loadingHistory ? (
                            <div></div>
                        ) : (
                            <>
                                {clientHistory.length === 0 ? (
                                    <p className='subtext' style={{ marginTop: '1rem' }}>Завершённых аренд ещё не было</p>
                                ) : (
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Оборудование</th>
                                                <th>Айтем</th>
                                                <th>Начало</th>
                                                <th>Конец</th>
                                                <th>Залог</th>
                                                <th>Штраф</th>
                                                <th>Сумма</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {clientHistory.map((record, index) => (
                                                <tr key={index}>
                                                    <td>{record.equipment}</td>
                                                    <td>{record.item}</td>
                                                    <td>{record.start_date}</td>
                                                    <td>{record.end_date}</td>
                                                    <td>{record.deposit}</td>
                                                    <td>{record.penalty}</td>
                                                    <td>{record.rent_sum}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Clients;
