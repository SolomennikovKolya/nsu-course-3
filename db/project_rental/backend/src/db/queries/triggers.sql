CREATE TRIGGER after_reservation_insert
AFTER INSERT ON Reservations
FOR EACH ROW
BEGIN
    -- Меняем статус первого доступного элемента на 'booked'
    UPDATE Items
    SET status = 'booked'
    WHERE equipment_id = NEW.equipment_id
    AND status = 'available'
    LIMIT 1;
END//

CREATE TRIGGER after_reservation_cancelled
AFTER UPDATE ON Reservations
FOR EACH ROW
BEGIN
    IF NEW.status = 'cancelled' THEN
        -- Меняем статус первого элемента с оборудованием на 'available'
        UPDATE Items
        SET status = 'available'
        WHERE equipment_id = NEW.equipment_id
        AND status = 'booked'
        LIMIT 1;
    END IF;
END//

CREATE TRIGGER after_reservation_completed
AFTER UPDATE ON Reservations
FOR EACH ROW
BEGIN
    IF NEW.status = 'completed' THEN
        -- Меняем статус первого элемента на 'rented'
        UPDATE Items
        SET status = 'rented'
        WHERE equipment_id = NEW.equipment_id
        AND status = 'booked'
        LIMIT 1;

        -- Добавляем запись в Rentals
        INSERT INTO Rentals (client_id, item_id, start_date, end_date, status)
        VALUES (NEW.client_id, 
                (SELECT id FROM Items WHERE equipment_id = NEW.equipment_id AND status = 'rented' LIMIT 1), 
                NEW.start_date, NEW.end_date, 'active');
    END IF;
END//
