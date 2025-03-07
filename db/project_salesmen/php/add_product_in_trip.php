<?php
require_once 'connection.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $trip_id = $_POST['trip_id'];
    $product_id = $_POST['product_id'];
    $quantity_taken = $_POST['quantity_taken'];

    try {
        $pdo->beginTransaction();

        // Проверяем, существует ли запись в таблице products_in_trip
        $stmt = $pdo->prepare("
            SELECT quantity_taken 
            FROM products_in_trip 
            WHERE trip_id = :trip_id AND product_id = :product_id
        ");
        $stmt->execute(['trip_id' => $trip_id, 'product_id' => $product_id]);
        $existingProduct = $stmt->fetch(PDO::FETCH_ASSOC);

        if ($existingProduct) {
            // Если запись существует, обновляем количество
            $new_quantity = $existingProduct['quantity_taken'] + $quantity_taken;
            $stmt = $pdo->prepare("
                UPDATE products_in_trip 
                SET quantity_taken = :quantity_taken 
                WHERE trip_id = :trip_id AND product_id = :product_id
            ");
            $stmt->execute([
                'quantity_taken' => $new_quantity,
                'trip_id' => $trip_id,
                'product_id' => $product_id,
            ]);

            // Также обновляем в products_returned
            $stmt = $pdo->prepare("
                UPDATE products_returned 
                SET quantity_returned = quantity_returned + :quantity_taken 
                WHERE trip_id = :trip_id AND product_id = :product_id
            ");
            $stmt->execute([
                'quantity_taken' => $quantity_taken,
                'trip_id' => $trip_id,
                'product_id' => $product_id,
            ]);
        } else {
            // Если запись не существует, создаем новую
            $stmt = $pdo->prepare("
                INSERT INTO products_in_trip (trip_id, product_id, quantity_taken)
                VALUES (:trip_id, :product_id, :quantity_taken)
            ");
            $stmt->execute([
                'trip_id' => $trip_id,
                'product_id' => $product_id,
                'quantity_taken' => $quantity_taken,
            ]);

            $stmt = $pdo->prepare("
                INSERT INTO products_returned (trip_id, product_id, quantity_returned)
                VALUES (:trip_id, :product_id, :quantity_returned)
            ");
            $stmt->execute([
                'trip_id' => $trip_id,
                'product_id' => $product_id,
                'quantity_returned' => $quantity_taken,
            ]);
        }

        $pdo->commit();

        echo "<script>window.location.href = 'full_commissioner.php?commissioner_id={$_POST['commissioner_id']}';</script>";
        exit;
    } catch (PDOException $e) {
        $pdo->rollBack();
        die("Ошибка при добавлении товара: " . $e->getMessage());
    }
}
