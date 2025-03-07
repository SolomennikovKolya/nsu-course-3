<?php
require_once 'connection.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $trip_id = $_POST['trip_id'];
    $product_id = $_POST['product_id'];
    $quantity_sold = $_POST['quantity_sold'];
    $commissioner_id = $_POST['commissioner_id'];

    try {
        // Начинаем транзакцию
        $pdo->beginTransaction();

        // Получаем количество взятых товаров и возвращенных товаров
        $stmt = $pdo->prepare("
            SELECT quantity_returned 
            FROM products_returned 
            WHERE trip_id = :trip_id AND product_id = :product_id
        ");
        $stmt->execute(['trip_id' => $trip_id, 'product_id' => $product_id]);
        $product_data = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$product_data) {
            die("Ошибка: Товар не найден в таблице командировки.");
        }

        // Расчет доступного для продажи количества (Взято - Возвращено)
        $quantity_available_for_sale = $product_data['quantity_returned'];

        // Проверяем, не продано ли больше, чем доступно для продажи
        if ($quantity_sold > $quantity_available_for_sale) {
            die("Ошибка: Продано больше, чем доступно для продажи.");
        }

        // Обновляем количество возвращенных товаров
        $stmt = $pdo->prepare("
            SELECT quantity_returned 
            FROM products_returned 
            WHERE trip_id = :trip_id AND product_id = :product_id
        ");
        $stmt->execute(['trip_id' => $trip_id, 'product_id' => $product_id]);
        $returned = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$returned) {
            die("Ошибка: Товар не найден в таблице возвращенных.");
        }

        // Проверяем, не продано ли больше, чем возвращено
        if ($quantity_sold > $returned['quantity_returned']) {
            die("Ошибка: Продано больше, чем возвращено товара.");
        }

        // Обновляем количество возвращенных товаров
        $new_quantity = $returned['quantity_returned'] - $quantity_sold;

        if ($new_quantity == 0) {
            // Устанавливаем количество возвращенного товара в 0, а не удаляем запись
            $stmt = $pdo->prepare("
                UPDATE products_returned 
                SET quantity_returned = 0 
                WHERE trip_id = :trip_id AND product_id = :product_id
            ");
            $stmt->execute(['trip_id' => $trip_id, 'product_id' => $product_id]);
        } else {
            // Обновляем запись в таблице продуктов возвращенных
            $stmt = $pdo->prepare("
                UPDATE products_returned 
                SET quantity_returned = :quantity_returned 
                WHERE trip_id = :trip_id AND product_id = :product_id
            ");
            $stmt->execute([
                'quantity_returned' => $new_quantity,
                'trip_id' => $trip_id,
                'product_id' => $product_id
            ]);
        }

        // Подтверждаем транзакцию
        $pdo->commit();

        // Перенаправляем обратно на страницу с командировкой
        echo "<script>window.location.href = 'full_commissioner.php?commissioner_id=$commissioner_id';</script>";
        exit;
    } catch (PDOException $e) {
        $pdo->rollBack();
        die("Ошибка при обновлении данных: " . $e->getMessage());
    }
}