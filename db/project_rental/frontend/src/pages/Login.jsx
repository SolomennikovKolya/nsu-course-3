import { useState } from 'react';
import { useAuth } from '../context/AuthContext';

function Login() {
    const [identifier, setIdentifier] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const { login } = useAuth();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await login(identifier, password);
        } catch {
            setError('Неверный логин или пароль');
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ padding: '2rem' }}>
            <h2>Вход</h2>
            <input placeholder="Email, имя или телефон" value={identifier} onChange={(e) => setIdentifier(e.target.value)} /><br />
            <input type="password" placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)} /><br />
            <button type="submit">Войти</button>
            {error && <p style={{ color: 'red' }}>{error}</p>}
        </form>
    );
}

export default Login;
