import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Home from './pages/Home';
import Login from './pages/Login';

// <Router> — обёртка для всего приложения, которая отслеживает изменения URL
// <Routes> — определяет область, где будут проверяться маршруты
// <Route path="/" element={<Home />} /> - Если пользователь заходит на корневой путь (или при входе на сайт), отображается компонент Home
// <Route path="/login" element={<Login />} /> - Если URL меняется на /login, отображается компонент Login

function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}
export default App;