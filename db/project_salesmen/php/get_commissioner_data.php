<?php
require_once 'connection.php';

$commissioner_id = $_GET['commissioner_id'] ?? null;
if (!$commissioner_id) {
    die("Ошибка: Не указан ID коммивояжера.");
}

$stmt = $pdo->prepare("SELECT * FROM commissioners WHERE id = :commissioner_id");
$stmt->execute(['commissioner_id' => $commissioner_id]);
$commissioner = $stmt->fetch(PDO::FETCH_ASSOC);
if (!$commissioner) {
    die("Ошибка: Коммивояжер не найден.");
}

$stmt = $pdo->prepare("SELECT * FROM business_trips WHERE commissioner_id = :commissioner_id ORDER BY start_date DESC LIMIT 1");
$stmt->execute(['commissioner_id' => $commissioner_id]);
$current_trip = $stmt->fetch(PDO::FETCH_ASSOC);

$products = $pdo->query("SELECT id, name, COALESCE(unit, 'шт') AS unit FROM products")->fetchAll(PDO::FETCH_ASSOC);

$current_trip_products = [];
if ($current_trip) {
    $stmt = $pdo->prepare("SELECT p.id AS product_id, p.name, p.price, COALESCE(p.unit, 'шт') AS unit, pit.quantity_taken, 
                    COALESCE(pr.quantity_returned, 0) AS quantity_sold 
                FROM products_in_trip pit
                LEFT JOIN products p ON pit.product_id = p.id
                LEFT JOIN products_returned pr ON pr.trip_id = pit.trip_id AND pr.product_id = pit.product_id
                WHERE pit.trip_id = :trip_id");
    $stmt->execute(['trip_id' => $current_trip['id']]);
    $current_trip_products = $stmt->fetchAll(PDO::FETCH_ASSOC);
}
