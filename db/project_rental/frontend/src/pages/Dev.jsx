import React from 'react';

function Dev() {
    const handleClearDB = () => {
        // Здесь будет логика очистки БД
        console.log('Очистка базы данных...');
    };

    const handleFillDB = () => {
        // Здесь будет логика заполнения БД
        console.log('Заполнение базы данных...');
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
