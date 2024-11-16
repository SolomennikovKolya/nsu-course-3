-- Таблица Коммивояжеров
CREATE TABLE IF NOT EXISTS commissioners (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    efficiency DECIMAL(10, 2) DEFAULT 0,
    address VARCHAR(255),
    phone VARCHAR(15) NOT NULL
);

-- Таблица Товаров с проверкой цены
CREATE TABLE IF NOT EXISTS products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price > 0),
    unit VARCHAR(50) NOT NULL CHECK (unit IN ('шт', 'кг'))
);

-- Таблица Командировок
CREATE TABLE IF NOT EXISTS business_trips (
    id INT AUTO_INCREMENT PRIMARY KEY,
    commissioner_id INT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    FOREIGN KEY (commissioner_id) REFERENCES commissioners(id)
);

-- Таблица Товары в командировке с проверкой количества
CREATE TABLE IF NOT EXISTS products_in_trip (
    trip_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity_taken DECIMAL(10,2) NOT NULL CHECK (quantity_taken > 0),
    FOREIGN KEY (trip_id) REFERENCES business_trips(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Таблица Возврат товаров с проверкой количества
CREATE TABLE IF NOT EXISTS products_returned (
    trip_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity_returned DECIMAL(10,2) NOT NULL CHECK (quantity_returned >= 0),
    FOREIGN KEY (trip_id) REFERENCES business_trips(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
