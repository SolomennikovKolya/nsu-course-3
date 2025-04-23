import { createContext, useContext, useEffect, useState } from 'react';
import axios from '../axios';
import { useNavigate } from 'react-router-dom';

const AuthContext = createContext();

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null); // { username, role }
    const navigate = useNavigate();

    // Проверка токена при монтировании
    useEffect(() => {
        const checkAuth = async () => {
            const token = localStorage.getItem('access_token');
            if (!token) return;

            try {
                const res = await axios.get('/protected', {
                    headers: { Authorization: `Bearer ${token}` }
                });
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
        setUser({ role: res.data.role });
        navigate('/');
    };

    const logout = async () => {
        try {
            await axios.post('/logout');
        } catch { }
        localStorage.removeItem('access_token');
        setUser(null);
        navigate('/');
    };

    return (
        <AuthContext.Provider value={{ user, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    return useContext(AuthContext);
}
