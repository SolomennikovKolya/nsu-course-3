import React, { useState } from 'react';
import { useAuth } from '../AuthContext';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';

function NavbarTop() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
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
                backgroundColor: location.pathname === '/login' ? '#ddd' : '#f0f0f0'
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
            <div style={{
                display: 'flex',
                alignItems: 'center',
                padding: '1rem',
                gap: '1rem',
                flexWrap: 'wrap'
            }}>
                {/* 🟡 Клиентские островки */}
                <div style={{
                    display: 'flex',
                    gap: '1rem',
                    padding: '0.5rem 1rem',
                    backgroundColor: '#f0f0f0',
                    borderRadius: '10px',
                    alignItems: 'center'
                }}>
                    <NavLink to="/terms" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}>📄Условия</NavLink>
                    <NavLink to="/company" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> ❓Компания </NavLink>
                    <NavLink to="/contacts" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> ☎️Контакты </NavLink>
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
                        <NavLink to="/rents" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> 🗝️Аренды </NavLink>
                        <NavLink to="/notifications" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> ⚠️Уведомления </NavLink>
                        <NavLink to="/clients" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> 🙋Клиенты </NavLink>
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
                        <NavLink to="/equipment" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}>💲Оборудование </NavLink>
                        <NavLink to="/reports" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> 📈Отчёты </NavLink>
                        <NavLink to="/employees" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> 🪪Сотрудники </NavLink>
                    </div>
                )}

                {/* ⚙️ Development отдельно справа */}
                <div style={{ marginLeft: 'auto' }}>
                    <NavLink to="/dev" className={({ isActive }) => isActive ? "link-style active-link" : "link-style"}> ⚙️Development </NavLink>
                </div>
            </div>
        </div>
    );
}

export default NavbarTop;
