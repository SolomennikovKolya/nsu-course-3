import React from 'react';
import { Outlet } from 'react-router-dom';
import NavbarTop from './NavbarTop';

// Обёртка для всего сайта, чтобы добавить вершнюю панель на каждую страницу
function Layout() {
    return (
        <div>
            <NavbarTop />
            <main>
                <Outlet />
            </main>
        </div>
    );
}

export default Layout;
