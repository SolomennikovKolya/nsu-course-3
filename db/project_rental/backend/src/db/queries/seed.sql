INSERT INTO Equipment (name, category, description, rental_price_per_day, penalty_per_day, deposit_amount) VALUES
('Перфоратор Bosch', 'Строительное', 'Мощный ударный перфоратор для бетона', 25, 5, 50),
('Бензопила Stihl', 'Садовое', 'Для обрезки деревьев', 30, 6, 60);

INSERT INTO Items (equipment_id, status, last_maintenance_date) VALUES
(1, 'available', '2024-12-01'),
(1, 'rented', '2025-01-10'),
(2, 'available', '2025-01-01');

INSERT INTO Users (user_role, password_hash, name, phone, email) VALUES
('client', NULL, 'Клиент Тестов', '+79130000001', 'client_testov@example.com'),
('client', NULL, 'Алексей Смирнов', '+79130000004', 'alexey.smirnov@example.com'),
('client', NULL, 'Мария Иванова', '+79130000005', 'maria.ivanova@example.com'),
('client', NULL, 'Дмитрий Кузнецов', '+79130000006', 'dmitry.kuznetsov@example.com'),
('client', NULL, 'Анна Попова', '+79130000007', 'anna.popova@example.com'),
('client', NULL, 'Иван Соколов', '+79130000008', 'ivan.sokolov@example.com'),
('client', NULL, 'Светлана Орлова', '+79130000009', 'svetlana.orlova@example.com'),
('client', NULL, 'Олег Морозов', '+79130000010', 'oleg.morozov@example.com'),
('manager', 'scrypt:32768:8:1$rT00M9IGcC5uoirK$b4492f385c873d2c07b211b460fe9b901f2b6f48b4f3ce43bbc9be862d74a24889de88929979cd665a80c994f499f0b485945eca8d868768015cf1d7d539dd4a', 'Менеджер Тестов', '+79130000002', 'manager_testov@example.com'),
('manager', 'testhash', 'Менеджер Васильев', '+79130000011', 'vasiliev.manager@example.com'),
('manager', 'testhash', 'Менеджер Петров', '+79130000012', 'petrov.manager@example.com'),
('admin', 'scrypt:32768:8:1$svSg3LOqcvdu0wHH$073124f741222bdfac3e73552387597b475ee086d9edce5b608273a2afe29f8114fd1b7c944760ac5695417471583be02f02abbdcd50cabdbb7b70d647028645', 'Админ Тестов', '+79130000003', 'admin_testov@example.com');

INSERT INTO Reservations (client_id, equipment_id, start_date, end_date, status) VALUES
(1, 1, '2025-04-05', '2025-04-07', 'active');

INSERT INTO Rentals (client_id, item_id, start_date, end_date, extended_end_date, actual_return_date, total_cost, deposit_paid, penalty_amount, status) VALUES
(3, 1, '2025-03-10', '2025-03-15', NULL, '2025-03-15', 125, 50, 0, 'completed'),
(4, 1, '2025-03-20', '2025-03-25', NULL, '2025-03-24', 150, 50, 0, 'completed'),
(5, 3, '2025-04-01', '2025-04-05', NULL, NULL, 120, 60, 0, 'active'),
(6, 2, '2025-04-10', '2025-04-12', NULL, NULL, 60, 30, 0, 'active'),
(7, 1, '2025-04-15', '2025-04-20', NULL, NULL, 100, 50, 0, 'active'),
(8, 3, '2025-04-18', '2025-04-22', NULL, NULL, 130, 60, 0, 'active');

INSERT INTO Notifications (manager_id, type, message, created_at, is_read) VALUES
(9, 'new booking', 'Новая бронь: Алексей Смирнов забронировал оборудование', NOW(), FALSE),
(9, 'rental expiration', 'Аренда Дмитрия Кузнецова заканчивается через 1 день', NOW(), TRUE),
(9, 'question', 'Анна Попова задала вопрос о доступности перфоратора', NOW(), FALSE),
(9, 'rental expiration', 'Истекает аренда Иван Соколов', NOW(), FALSE),
(10, 'new booking', 'Новая бронь: Мария Иванова арендует бензопилу', NOW(), FALSE),
(10, 'new booking', 'Светлана Орлова оформила бронь на перфоратор', NOW(), FALSE),
(10, 'question', 'Олег Морозов спрашивает о продлении аренды бензопилы', NOW(), TRUE),
(10, 'rental expiration', 'Скоро окончание аренды Светланы Орловой', NOW(), FALSE),
(11, 'new booking', 'Олег Морозов оформил новую аренду', NOW(), FALSE),
(11, 'question', 'Мария Иванова уточняет условия оплаты', NOW(), FALSE),
(11, 'rental expiration', 'Истекает срок аренды оборудования у Алексея Смирнова', NOW(), FALSE);
