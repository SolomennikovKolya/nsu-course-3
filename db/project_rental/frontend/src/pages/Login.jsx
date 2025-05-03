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
            <form onSubmit={handleSubmit} style={{ padding: '2rem' }}>
                <h1 className="page-title">Вход</h1>
                <div style={{ width: 400 }}>
                    <input className="text-input" placeholder="Имя / Телефон / Email" value={identifier} onChange={(e) => setIdentifier(e.target.value)} />
                    <input className="text-input" type="password" placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)} />
                    <button type="submit" className="action-button">Войти</button>
                    {error && <p style={{ color: 'red' }}>{error}</p>}
                </div>
            </form>
        </div>

    );
}

export default Login;
