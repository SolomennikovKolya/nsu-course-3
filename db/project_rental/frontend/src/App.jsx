import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './AuthContext';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dev from './pages/Dev';
import Catalog from './pages/client/Catalog';
import Category from './pages/client/Category';
import EquipmentCard from './pages/client/EquipmentCard';
import Terms from './pages/client/Terms';
import Company from './pages/client/Company';
import Contacts from './pages/client/Contacts';
import Clients from './pages/manager/Clients';
import Employees from './pages/admin/Employees';
import Equipment from './pages/admin/Equipment';

// <Router> - Обёртка для всего приложения, которая отслеживает изменения URL
// <AuthProvider> - Даёт доступ к состоянию авторизации (user, login(), logout()) через хук useAuth()
// <Routes> - Контейнер для всех маршрутов (<Route>)
// <Route> - Определяет, какой компонент показывать при заданном пути
function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<Layout />}>
            {/* Общие страницы */}
            <Route index element={<Catalog />} />
            <Route path="/login" element={<Login />} />
            <Route path="/dev" element={<Dev />} />

            {/* Страницы для клиента (тоже общие) */}
            <Route path="/catalog" element={<Catalog />} />
            <Route path="/catalog/:categoryNameSlug" element={<Category />} />
            <Route path="/catalog/:categoryNameSlug/:equipmentNameSlug" element={<EquipmentCard />} />
            <Route path="/terms" element={<Terms />} />
            <Route path="/company" element={<Company />} />
            <Route path="/contacts" element={<Contacts />} />

            {/* Страницы для менеджера */}
            <Route path="/clients" element={<Clients />} />

            {/* Страницы для админов */}
            <Route path="/equipment" element={<Equipment />} />
            <Route path="/employees" element={<Employees />} />
          </Route>
        </Routes>
      </AuthProvider>
    </Router >
  );
}

export default App;
