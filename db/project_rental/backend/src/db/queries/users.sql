DROP USER 'equipment_rental_client'@'localhost';
CREATE USER 'equipment_rental_client'@'localhost' IDENTIFIED BY 'client';

GRANT SELECT ON equipment_rental.Equipment TO 'equipment_rental_client'@'localhost';
GRANT SELECT ON equipment_rental.Items TO 'equipment_rental_client'@'localhost';
GRANT SELECT, INSERT ON equipment_rental.Users TO 'equipment_rental_client'@'localhost';
GRANT INSERT ON equipment_rental.Reservations TO 'equipment_rental_client'@'localhost';

FLUSH PRIVILEGES;

DROP USER 'equipment_rental_manager'@'localhost';
CREATE USER 'equipment_rental_manager'@'localhost' IDENTIFIED BY 'manager';

GRANT SELECT ON equipment_rental.Equipment TO 'equipment_rental_manager'@'localhost';
GRANT SELECT, INSERT ON equipment_rental.Users TO 'equipment_rental_manager'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON equipment_rental.Reservations TO 'equipment_rental_manager'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON equipment_rental.Rentals TO 'equipment_rental_manager'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON equipment_rental.Items TO 'equipment_rental_manager'@'localhost';

FLUSH PRIVILEGES;
