CREATE TABLE IF NOT EXISTS Equipment (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(255) NOT NULL,
    description TEXT,
    rental_price_per_day INT NOT NULL,
    penalty_per_day INT NOT NULL,
    deposit_amount INT NOT NULL
);

CREATE TABLE IF NOT EXISTS Items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    status ENUM('available', 'booked', 'rented', 'serviced', 'decommissioned') NOT NULL,
    last_maintenance_date DATE,
    FOREIGN KEY (equipment_id) REFERENCES Equipment(id)
);

CREATE TABLE IF NOT EXISTS Users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    role ENUM('client', 'manager', 'administrator') NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(50),
    email VARCHAR(255),
    registration_date DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS Reservations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    client_id INT NOT NULL,
    equipment_id INT NOT NULL,
    reservation_date DATETIME NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status ENUM('active', 'cancelled', 'completed') NOT NULL,
    FOREIGN KEY (client_id) REFERENCES Users(id),
    FOREIGN KEY (equipment_id) REFERENCES Equipment(id)
);

CREATE TABLE IF NOT EXISTS Rentals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    client_id INT NOT NULL,
    item_id INT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    extended_end_date DATE,
    actual_return_date DATE,
    total_cost INT,
    deposit_paid INT,
    penalty_amount INT,
    status ENUM('active', 'completed') NOT NULL,
    FOREIGN KEY (client_id) REFERENCES Users(id),
    FOREIGN KEY (item_id) REFERENCES Items(id)
);

CREATE TABLE IF NOT EXISTS Notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    manager_id INT NOT NULL,
    type ENUM('question', 'new booking', 'rental expiration') NOT NULL,
    message TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (manager_id) REFERENCES Users(id)
);
