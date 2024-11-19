<!-- Подключение к базе данных MySQL с использованием сохраненных в сессии данных пользователя -->

<?php
session_start();

try {
    $username = $_SESSION['db_username'];
    $password = $_SESSION['db_password'];

    if (!$username) {
        throw new Exception("Имя пользователя не установлено");
    }

    $dsn = 'mysql:host=localhost;dbname=project7;charset=utf8';
    $pdo = new PDO($dsn, $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

} catch (Exception $e) {
    echo "Ошибка подключения: " . $e->getMessage();
    exit;
}
