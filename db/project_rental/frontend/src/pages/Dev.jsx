import React from 'react';
import axios from '../axios';

function Dev() {
    const handleClearDB = async () => {
        try {
            const res = await axios.post('/clear_db');
            console.log('Ответ от сервера:', res.data);
            alert('База данных успешно очищена');
        } catch (err) {
            console.error('Ошибка при очистке базы данных:', err);
            alert('Ошибка при очистке базы данных');
        }
    };

    const handleFillDB = async () => {
        try {
            const res = await axios.post('/seed_db');
            console.log('Ответ от сервера:', res.data);
            alert('База данных успешно заполнена');
        } catch (err) {
            console.error('Ошибка при заполнении базы данных:', err);
            alert('Ошибка при заполнении базы данных');
        }
    };

    return (
        <div style={{ padding: '0 2rem 2rem 2rem' }}>
            <h1 className="page-title">Develop</h1>

            <p className="subtext">Управление базой данных</p>

            <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem' }}>
                <button onClick={handleClearDB} className="action-button">Очистить БД</button>
                <button onClick={handleFillDB} className="action-button">Заполнить БД</button>
            </div>
        </div>
    );
}

export default Dev;
