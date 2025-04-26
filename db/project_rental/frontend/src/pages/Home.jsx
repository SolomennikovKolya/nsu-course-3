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
            {/* Верхняя панель */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '1rem',
                backgroundColor: '#f0f0f0'
            }}>
                <div style={{ fontWeight: 'bold' }}>Аренда Оборудования</div>
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
                <button onClick={handleAuthClick}>
                    {user ? 'Выйти' : 'Войти'}
                </button>
            </div>

            {/* Панель всех кнопок */}
            <div style={{ display: 'flex', alignItems: 'center', padding: '1rem', gap: '1rem' }}>

                {/* 🟡 Клиентские островки */}
                <div style={{
                    display: 'flex',
                    gap: '1rem',
                    padding: '0.5rem 1rem',
                    backgroundColor: '#f0f0f0',
                    borderRadius: '10px',
                    alignItems: 'center'
                }}>
                    <Link to="/terms" className="link-style">📄Условия</Link>
                    <Link to="/company" className="link-style">❓Компания</Link>
                    <Link to="/contacts" className="link-style">☎️Контакты</Link>
                </div>

                {/* 🟠 Менеджерские островки */}
                {(user?.role === 'manager' || user?.role === 'admin') && (
                    <div style={{
                        display: 'flex',
                        gap: '1rem',
                        padding: '0.5rem 1rem',
                        backgroundColor: '#f0f0f0',
                        borderRadius: '10px',
                        alignItems: 'center'
                    }}>
                        <Link to="/rents" className="link-style">🗝️Аренды</Link>
                        <Link to="/notifications" className="link-style">⚠️Уведомления</Link>
                        <Link to="/clients" className="link-style">🙋Клиенты</Link>
                    </div>
                )}

                {/* 🔴 Админские островки */}
                {user?.role === 'admin' && (
                    <div style={{
                        display: 'flex',
                        gap: '1rem',
                        padding: '0.5rem 1rem',
                        backgroundColor: '#f0f0f0',
                        borderRadius: '10px',
                        alignItems: 'center'
                    }}>
                        <Link to="/equipment" className="link-style">💲Оборудование</Link>
                        <Link to="/reports" className="link-style">📈Отчёты</Link>
                        <Link to="/employees" className="link-style">🪪Сотрудники</Link>
                    </div>
                )}

                {/* ⚙️ Development отдельно справа */}
                <div style={{ marginLeft: 'auto' }}>
                    <Link to="/dev" className="link-style">⚙️Development</Link>
                </div>
            </div>
        </div>
    );
}

export default Home;