import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './AuthContext';
import Layout from './components/Layout';
import Home from './pages/Home';
import Login from './pages/Login';
import Dev from './pages/Dev';
import Terms from './pages/Terms';
import Company from './pages/Company';
import Contacts from './pages/Contacts';
import Clients from './pages/Clients';
import Catalog from './pages/Catalog';
import Category from './pages/Category';
import EquipmentCard from './pages/EquipmentCard';
import Employees from './pages/Employees';
import Equipment from './pages/Equipment';

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
            <Route index element={<Catalog />} />
            <Route path="/login" element={<Login />} />
            <Route path="/dev" element={<Dev />} />

            <Route path="/catalog" element={<Catalog />} />
            <Route path="/catalog/:categoryNameSlug" element={<Category />} />
            <Route path="/catalog/:categoryNameSlug/:equipmentNameSlug" element={<EquipmentCard />} />

            <Route path="/terms" element={<Terms />} />
            <Route path="/company" element={<Company />} />
            <Route path="/contacts" element={<Contacts />} />

            <Route path="/clients" element={<Clients />} />

            <Route path="/equipment" element={<Equipment />} />
            <Route path="/employees" element={<Employees />} />
          </Route>
        </Routes>
      </AuthProvider>
    </Router >
  );
}

export default App;
