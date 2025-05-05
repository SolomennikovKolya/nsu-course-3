-- При добавлении брони, статус первого доступного айтема (оборудования, указанного в брони) меняем на 'booked'
CREATE TRIGGER after_reservation_insert
AFTER INSERT ON Reservations
FOR EACH ROW
BEGIN
    UPDATE Items
    SET status = 'booked'
    WHERE equipment_id = NEW.equipment_id
    AND status = 'available'
    LIMIT 1;
END//

-- При отмени брони, статус первого забронированного айтема (оборудования, указанного в брони) меняем на 'available'
CREATE TRIGGER after_reservation_cancelled
AFTER UPDATE ON Reservations
FOR EACH ROW
BEGIN
    IF NEW.status = 'cancelled' THEN
        UPDATE Items
        SET status = 'available'
        WHERE equipment_id = NEW.equipment_id
        AND status = 'booked'
        LIMIT 1;
    END IF;
END//

-- При завершении брони (то есть перехода брони в аренду): 
-- 1. Находится первый айтем оборудования, указанного в брони, со статусом 'booked'
-- 2. Статус найденного айтема меняется на 'rented'
-- 3. Добавляется соответствующая аренда
CREATE TRIGGER after_reservation_completed
AFTER UPDATE ON Reservations
FOR EACH ROW
BEGIN
    DECLARE rented_item_id INT;
    DECLARE deposit INT;

    IF NEW.status = 'completed' THEN
        SELECT id INTO rented_item_id
        FROM Items
        WHERE equipment_id = NEW.equipment_id
        AND status = 'booked'
        LIMIT 1;

        IF rented_item_id IS NOT NULL THEN
            UPDATE Items
            SET status = 'rented'
            WHERE id = rented_item_id;

            SELECT deposit_amount INTO deposit
            FROM Equipment
            WHERE id = NEW.equipment_id;

            INSERT INTO Rentals (client_id, item_id, start_date, end_date, status, deposit_paid)
            VALUES (NEW.client_id, rented_item_id, NEW.start_date, NEW.end_date, 'active', COALESCE(deposit, 0));
        END IF;
    END IF;
END//
