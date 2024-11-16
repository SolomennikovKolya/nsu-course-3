<?php
session_start();

$username = $_POST['username'];
$password = $_POST['password'];
$role = $_POST['role'];
$commissioner_id = $_POST['commissioner_id'] ?? null;

$_SESSION['db_username'] = $username;
$_SESSION['db_password'] = $password;

try {
    $dsn = 'mysql:host=localhost;charset=utf8';
    $pdo = new PDO($dsn, $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // Создаем базу данных, если не существует
    $pdo->exec("CREATE DATABASE IF NOT EXISTS project7");

    // Подключаемся к базе данных project7
    $dsn = 'mysql:host=localhost;dbname=project7;charset=utf8';
    $pdo = new PDO($dsn, $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // Выполняем удаление таблиц
    $dropFile = '../sql/drop_tables.sql';
    if (file_exists($dropFile)) {
        $dropSql = file_get_contents($dropFile);
        $pdo->exec($dropSql);
    } else {
        throw new Exception("Файл drop_tables.sql не найден.");
    }

    // Загружаем и выполняем схему
    $schemaFile = '../sql/schema.sql';
    if (file_exists($schemaFile)) {
        $schema = file_get_contents($schemaFile);
        $pdo->exec($schema);
    } else {
        throw new Exception("Файл schema.sql не найден.");
    }

    // Вставляем тестовые данные
    $testDataFile = '../sql/test_data.sql';
    if (file_exists($testDataFile)) {
        $testData = file_get_contents($testDataFile);
        $pdo->exec($testData);
    } else {
        throw new Exception("Файл test_data.sql не найден.");
    }

    if ($role == 'admin') {
        echo "<script>window.location.href = '../home.php';</script>";
        exit();
    } elseif ($role == 'commissioner' && $commissioner_id) {
        echo "<script>window.location.href = 'full_commissioner.php?commissioner_id=" . $commissioner_id . "';</script>";
        exit();
    } else {
        echo "Неверная роль или не указан ID коммивояжера";
    }

    exit;
} catch (PDOException $e) {
    echo "Ошибка подключения к базе данных: " . $e->getMessage();
    exit;
} catch (Exception $e) {
    echo "Ошибка: " . $e->getMessage();
    exit;
}
