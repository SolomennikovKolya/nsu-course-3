INSERT INTO Equipment (name, category, description, rental_price_per_day, penalty_per_day, deposit_amount) VALUES
('Перфоратор Bosch', 'Строительное', 'Мощный ударный перфоратор для бетона', 25.00, 5.00, 50.00),
('Бензопила Stihl', 'Садовое', 'Для обрезки деревьев', 30.00, 6.00, 60.00);

INSERT INTO Items (equipment_id, status, last_maintenance_date) VALUES
(1, 'available', '2024-12-01'),
(1, 'rented', '2025-01-10'),
(2, 'available', '2025-01-01');

INSERT INTO Users (role, password_hash, name, phone, email, registration_date) VALUES
('client', 'scrypt:32768:8:1$57T6bS5tBgYEBMh9$37843a1ec4153393b65ed5cb42a47bd6e980600c7d4b8c1319259faa3bc7a6d3f864af6478b0b9efe879d8668b4bc7092484b4961f49818c37e3ce968ca030e0', 'Клиент Тестов', '+79130000001', 'client_testov@example.com', '2025-01-01'),
('manager', 'scrypt:32768:8:1$rT00M9IGcC5uoirK$b4492f385c873d2c07b211b460fe9b901f2b6f48b4f3ce43bbc9be862d74a24889de88929979cd665a80c994f499f0b485945eca8d868768015cf1d7d539dd4a', 'Менеджер Тестов', '+79130000002', 'manager_testov@example.com', '2025-01-02'),
('admin', 'scrypt:32768:8:1$svSg3LOqcvdu0wHH$073124f741222bdfac3e73552387597b475ee086d9edce5b608273a2afe29f8114fd1b7c944760ac5695417471583be02f02abbdcd50cabdbb7b70d647028645', 'Админ Тестов', '+79130000003', 'admin_testov@example.com', '2025-01-3');

INSERT INTO Reservations (client_id, equipment_id, reservation_date, start_date, end_date, status) VALUES
(1, 1, '2025-04-01', '2025-04-05', '2025-04-07', 'active');

INSERT INTO Rentals (client_id, item_id, start_date, end_date, extended_end_date, actual_return_date, total_cost, deposit_paid, penalty_amount, status) VALUES
(1, 2, '2025-03-01', '2025-03-05', NULL, '2025-03-05', 100.00, 50.00, 0.00, 'completed');

INSERT INTO Notifications (manager_id, type, message, created_at, is_read) VALUES
(2, 'new booking', 'Новая бронь от клиента Иван Клиент', NOW(), FALSE),
(2, 'rental expiration', 'Истекает аренда перфоратора Bosch', NOW(), FALSE);
