import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';

function Login() {
    const [identifier, setIdentifier] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await login(identifier, password);
        } catch {
            setError('Неверный логин или пароль');
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit} style={{ padding: '0 2rem 2rem 2rem' }}>
                <h1 className="page-title">Вход</h1>
                <input className="text-input" placeholder="Имя / Телефон / Email" value={identifier} onChange={(e) => setIdentifier(e.target.value)} /><br />
                <input className="text-input" type="password" placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)} /><br />
                <button type="submit" className="action-button">Войти</button>
                {error && <p style={{ color: 'red' }}>{error}</p>}
            </form>
        </div>

    );
}

export default Login;
