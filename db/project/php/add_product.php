<?php
require_once 'connection.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = $_POST['name'];
    $price = $_POST['price'];
    $unit = $_POST['unit'];

    $sql = "INSERT INTO products (name, price, unit) VALUES (?, ?, ?)";
    $stmt = $pdo->prepare($sql);

    $stmt->execute([$name, $price, $unit]);

    echo "<script>window.location.href = '../home.php';</script>";
}
