import { createContext, useContext, useEffect, useState } from 'react';
import axios from './axios';
import { useNavigate } from 'react-router-dom';

// Создаётся контекст - по сути, глобальная переменная, доступная из любого компонента через useAuth()
const AuthContext = createContext();

// Компонент-обёртка, который даёт всем детям доступ к переменной user 
// (текущая роль или null, если не авторизован) и функциям login, logout
export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const navigate = useNavigate();

    // Проверка токена при монтировании
    // Нужно, чтобы восстановить авторизацию, если пользователь уже вошёл ранее и у него остался токен
    useEffect(() => {
        const checkAuth = async () => {
            const token = localStorage.getItem('access_token');
            if (!token) return;

            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

            try {
                const res = await axios.get('/protected');
                setUser(res.data);
            } catch {
                setUser(null);
            }
        };
        checkAuth();
    }, []);

    const login = async (identifier, password) => {
        const res = await axios.post('/login', { identifier, password });

        localStorage.setItem('access_token', res.data.access_token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        setUser({ role: res.data.role });

        navigate('/');
    };

    const logout = async () => {
        try {
            await axios.post('/logout');
        } catch { }

        localStorage.removeItem('access_token');
        axios.defaults.headers.common['Authorization'] = null;
        setUser(null);

        navigate('/');
    };

    return (
        <AuthContext.Provider value={{ user, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

// Удобный способ использовать контекст в любом компоненте:
// const { user, login, logout } = useAuth();
export function useAuth() {
    return useContext(AuthContext);
}
