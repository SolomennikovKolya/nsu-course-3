import React, { useState } from 'react';
import { useAuth } from '../AuthContext';
import { Link, useNavigate } from 'react-router-dom';

function Home() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [search, setSearch] = useState('');

    const clearSearch = () => setSearch('');
    const handleAuthClick = () => {
        if (user) logout();
        else navigate('/login');
    };

    return (
        <div>
            {/* 🔵 Верхняя панель */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '1rem',
                backgroundColor: '#eee'
            }}>
                <div style={{ fontWeight: 'bold' }}>🏢 Название компании</div>

                <button>Каталог</button>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                        type="text"
                        placeholder="Поиск..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                    />
                    <button onClick={clearSearch}>X</button>
                    <button>Поиск</button>
                </div>

                {/* 🔐 Войти / Выйти */}
                <button onClick={handleAuthClick}>
                    {user ? 'Выйти' : 'Войти'}
                </button>
            </div>

            {/* 🟢 Общая панель */}
            <div style={{
                padding: '1rem',
                backgroundColor: '#f8f8f8',
                display: 'flex',
                gap: '1rem',
                justifyContent: 'space-between',
                flexWrap: 'wrap'
            }}>
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <Link to="/terms">Условия</Link>
                    <Link to="/company">Компания</Link>
                    <Link to="/contacts">Контакты</Link>
                </div>
                <div style={{ marginLeft: 'auto' }}>
                    <Link to="/dev">Development</Link>
                </div>
            </div>

            {/* 🟡 Менеджеры и админы */}
            {(user?.role === 'manager' || user?.role === 'admin') && (
                <div style={{ padding: '1rem', backgroundColor: '#ddd', display: 'flex', gap: '1rem' }}>
                    <Link to="/rents">Аренды</Link>
                    <Link to="/notifications">Уведомления</Link>
                    <Link to="/clients">Клиенты</Link>
                </div>
            )}

            {/* 🔴 Только админы */}
            {user?.role === 'admin' && (
                <div style={{ padding: '1rem', backgroundColor: '#ccc', display: 'flex', gap: '1rem' }}>
                    <Link to="/equipment">Оборудование</Link>
                    <Link to="/reports">Отчёты</Link>
                    <Link to="/employees">Сотрудники</Link>
                </div>
            )}
        </div>
    );
}

export default Home;
