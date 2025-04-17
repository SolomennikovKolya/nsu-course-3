import { useNavigate } from 'react-router-dom';

function Home() {
    const navigate = useNavigate();
    return (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
            <h1>Добро пожаловать в SPA</h1>
            <button onClick={() => navigate('/login')}>Войти</button>
        </div>
    );
}
export default Home;
