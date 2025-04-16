import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import AdminPage from './pages/Admin';
import ManagerPage from './pages/Manager';
import UserPage from './pages/User';
import Login from './components/Login';

function App() {
    const [user, setUser] = useState(null);

    const handleLogin = async (username, password) => {
        const response = await fetch('http://localhost:5000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();
        if (data.success) {
            setUser(data.user);
            return true;
        }
        return false;
    };

    const handleLogout = () => {
        setUser(null);
    };

    return (
        <Router>
            <div className="App">
                {!user ? (
                    <Login onLogin={handleLogin} />
                ) : (
                    <>
                        <button onClick={handleLogout}>Logout</button>
                        <Routes>
                            <Route path="/admin" element={
                                user.role === 'admin' ? <AdminPage user={user} /> : <Navigate to={`/${user.role}`} />
                            } />
                            <Route path="/manager" element={
                                user.role === 'manager' ? <ManagerPage user={user} /> : <Navigate to={`/${user.role}`} />
                            } />
                            <Route path="/user" element={<UserPage user={user} />} />
                            <Route path="*" element={<Navigate to={`/${user.role}`} />} />
                        </Routes>
                    </>
                )}
            </div>
        </Router>
    );
}

export default App;
