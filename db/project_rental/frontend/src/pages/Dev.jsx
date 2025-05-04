import React from 'react';
import axios from '../axios';

function Dev() {
    const handleInitDB = async () => {
        try {
            const res = await axios.post('/dev/init_db');
            console.log('Ответ от сервера:', res.data);
            alert('База данных успешно создана');
        } catch (err) {
            console.error('Ошибка при создании базы данных:', err);
            alert('Ошибка при создании базы данных');
        }
    };

    const handleSeedDB = async () => {
        try {
            const res = await axios.post('/dev/seed_db');
            console.log('Ответ от сервера:', res.data);
            alert('База данных успешно заполнена');
        } catch (err) {
            console.error('Ошибка при заполнении базы данных:', err);
            alert('Ошибка при заполнении базы данных');
        }
    };

    const handleClearDB = async () => {
        try {
            const res = await axios.post('/dev/clear_db');
            console.log('Ответ от сервера:', res.data);
            alert('База данных успешно очищена');
        } catch (err) {
            console.error('Ошибка при очистке базы данных:', err);
            alert('Ошибка при очистке базы данных');
        }
    };

    const handleDropDB = async () => {
        try {
            const res = await axios.post('/dev/drop_db');
            console.log('Ответ от сервера:', res.data);
            alert('База данных успешно удалена');
        } catch (err) {
            console.error('Ошибка при удалении базы данных:', err);
            alert('Ошибка при удалении базы данных');
        }
    };

    return (
        <div style={{ padding: '2rem' }}>
            <h1 className="page-title">Development</h1>
            <p className="subtext">Управление базой данных</p>

            <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem' }}>
                <button onClick={handleInitDB} className="gray-button">Создать БД</button>
                <button onClick={handleSeedDB} className="gray-button">Заполнить БД</button>
                <button onClick={handleClearDB} className="gray-button">Очистить БД</button>
                <button onClick={handleDropDB} className="gray-button">Удалить БД</button>
            </div>
        </div>
    );
}

export default Dev;
