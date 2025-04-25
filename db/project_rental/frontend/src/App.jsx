import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './AuthContext';
import Home from './pages/Home';
import Login from './pages/Login';
import './main.css';

// <Router> - Обёртка для всего приложения, которая отслеживает изменения URL
// <AuthProvider> - Даёт доступ к состоянию авторизации (user, login(), logout()) через хук useAuth()
// <Routes> - Контейнер для всех <Route>
// <Route> - Определяет, какой компонент показывать при заданном пути
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
