<!-- Первичное подключение к БД. Пересоздаёт БД, если с ней что то не так. В конце перенаправляет на страницу админа или коммивояжера -->

<?php
// Начинается сессия для текущего пользователя. Это позволяет сохранять данные (например, имя пользователя и пароль) между запросами.
session_start();

// Получает данные, отправленные пользователем через форму
$username = $_POST['username'];
$password = $_POST['password'];
$role = $_POST['role'];
$commissioner_id = $_POST['commissioner_id'] ?? null;

// Сохранение данных в сессии (имя пользователя и пароль), чтобы использовать их в других частях сайта
$_SESSION['db_username'] = $username;
$_SESSION['db_password'] = $password;

try {
    // Подключение к серверу MySQL
    $dsn = 'mysql:host=localhost;charset=utf8'; // Строка подключения с указанием хоста (localhost) и кодировки (utf8).
    $pdo = new PDO($dsn, $username, $password); // Создает объект подключения к базе данных (PDO / PHP data object).
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION); // Устанавливает режим ошибок (генерация исключений при ошибках).

    // Создаем базу данных, если она не существует. Метод exec выполняет SQL-запрос, который изменяет данные в базе
    $pdo->exec("CREATE DATABASE IF NOT EXISTS project7");

    // Подключение (обновление подключения) к базе данных project7.
    $dsn = 'mysql:host=localhost;dbname=project7;charset=utf8';
    $pdo = new PDO($dsn, $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // Проверка существования всех таблиц
    $tables = ['products_returned', 'products_in_trip', 'business_trips', 'products', 'commissioners'];
    $allTablesExist = true;
    foreach ($tables as $table) {
        // Метод query() выполняет SQL-запрос на сервере базы данных и возвращает объект, который содержит результат выполнения запроса.
        // SHOW TABLES — это SQL-команда, которая возвращает список всех таблиц в текущей базе данных.
        // LIKE '$table' — это условие фильтрации, которое ищет таблицу, имя которой совпадает с значением переменной $table. 
        // Это позволяет проверить, существует ли таблица с таким именем в базе данных.
        // Метод fetch() извлекает одну строку результата запроса. Если таблица не найдена, то метод fetch() вернет false.
        $result = $pdo->query("SHOW TABLES LIKE '$table'")->fetch();
        if (!$result) {
            $allTablesExist = false;
            break;
        }
    }

    // Если хотя бы одна таблица не существует, удаляем все таблицы и перезаписываем их
    if (!$allTablesExist) {
        // Удаляем таблицы, если они существуют
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
    }

    // Редирект в зависимости от роли
    if ($role == 'admin') {
        echo "<script>window.location.href = '../home.php';</script>";
        exit();
    } elseif ($role == 'commissioner' && $commissioner_id) {
        // echo — это команда в PHP, которая выводит строку на экран. В данном случае выводится HTML-код, который будет интерпретироваться браузером.
        // <script>...</script> — это тег HTML для вставки JavaScript кода на страницу
        // window.location.href — это JavaScript свойство, которое позволяет изменять текущий URL-адрес в адресной строке браузера.
        // commissioner_id= — это параметр в URL. После знака вопроса (?) идут параметры (в данном случае один параметр commissioner_id) перечисленные через амперсанд (&) в случае нескольких параметров.
        echo "<script>window.location.href = 'full_commissioner.php?commissioner_id=" . $commissioner_id . "';</script>";
        exit();
    } else {
        echo "Неверная роль или не указан ID коммивояжера";
    }

    // Завершение выполнения скрипта
    exit;

} catch (PDOException $e) {
    echo "Ошибка подключения к базе данных: " . $e->getMessage();
    exit;

} catch (Exception $e) {
    echo "Ошибка: " . $e->getMessage();
    exit;
}
