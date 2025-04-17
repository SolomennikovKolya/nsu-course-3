import { useState } from 'react';
import axios from 'axios';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const res = await axios.post('http://localhost:5000/api/auth/login', { username, password });
            alert(`Добро пожаловать, ${res.data.username} (роль: ${res.data.role})`);
        } catch (err) {
            setError('Неверный логин или пароль');
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ padding: '2rem' }}>
            <h2>Вход</h2>
            <input placeholder="Логин" value={username} onChange={(e) => setUsername(e.target.value)} /><br />
            <input type="password" placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)} /><br />
            <button type="submit">Войти</button>
            {error && <p style={{ color: 'red' }}>{error}</p>}
        </form>
    );
}
export default Login;