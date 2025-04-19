INSERT INTO Equipment (name, category, description, rental_price_per_day, penalty_per_day, deposit_amount) VALUES
('Перфоратор Bosch', 'Строительное', 'Мощный ударный перфоратор для бетона', 25.00, 5.00, 50.00),
('Бензопила Stihl', 'Садовое', 'Для обрезки деревьев', 30.00, 6.00, 60.00);

INSERT INTO Items (equipment_id, status, last_maintenance_date) VALUES
(1, 'available', '2024-12-01'),
(1, 'rented', '2025-01-10'),
(2, 'available', '2025-01-01');

INSERT INTO Users (role, password, name, phone, email, registration_date) VALUES
('client', 'pass123', 'Иван Клиент', '+79001234567', 'ivan@example.com', '2025-01-01'),
('manager', 'pass456', 'Мария Менеджер', '+79007654321', 'maria@example.com', '2025-01-05'),
('administrator', 'adminpass', 'Админ Администраторов', NULL, 'admin@example.com', '2025-01-10');

INSERT INTO Reservations (client_id, equipment_id, reservation_date, start_date, end_date, status) VALUES
(1, 1, '2025-04-01', '2025-04-05', '2025-04-07', 'active');

INSERT INTO Rentals (client_id, item_id, start_date, end_date, extended_end_date, actual_return_date, total_cost, deposit_paid, penalty_amount, status) VALUES
(1, 2, '2025-03-01', '2025-03-05', NULL, '2025-03-05', 100.00, 50.00, 0.00, 'completed');

INSERT INTO Notifications (manager_id, type, message, created_at, is_read) VALUES
(2, 'new booking', 'Новая бронь от клиента Иван Клиент', NOW(), FALSE),
(2, 'rental expiration', 'Истекает аренда перфоратора Bosch', NOW(), FALSE);
