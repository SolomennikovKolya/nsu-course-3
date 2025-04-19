import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

function Home() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    return (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
            <h1>Добро пожаловать в SPA</h1>
            <p>Ваша роль: <strong>{user?.role || 'client'}</strong></p>
            {user ? (
                <button onClick={logout}>Выйти</button>
            ) : (
                <button onClick={() => navigate('/login')}>Войти</button>
            )}
        </div>
    );
}

export default Home;
