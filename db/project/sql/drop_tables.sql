-- Отключение проверки внешних ключей, чтобы не вылезали ошибки связанные с нарушением целостности ссылок
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS products_returned;
DROP TABLE IF EXISTS products_in_trip;
DROP TABLE IF EXISTS business_trips;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS commissioners;

SET FOREIGN_KEY_CHECKS = 1;