<?php
require_once 'connection.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $full_name = $_POST['full_name'];
    $address = $_POST['address'];
    $phone = $_POST['phone'];

    $stmt = $pdo->prepare("INSERT INTO commissioners (full_name, address, phone) VALUES (?, ?, ?)");
    $stmt->execute([$full_name, $address, $phone]);

    echo "<script>window.location.href = '../home.php';</script>";
}
