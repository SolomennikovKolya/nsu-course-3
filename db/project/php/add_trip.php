<?php
require_once 'connection.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $commissioner_id = $_POST['commissioner_id'];
    $start_date = $_POST['start_date'];
    $end_date = $_POST['end_date'];

    $stmt = $pdo->prepare("SELECT COUNT(*) FROM business_trips WHERE commissioner_id = :commissioner_id AND start_date = :start_date AND end_date = :end_date");
    $stmt->execute(['commissioner_id' => $commissioner_id, 'start_date' => $start_date, 'end_date' => $end_date]);
    $count = $stmt->fetchColumn();

    if ($count == 0) {

        $sql = "INSERT INTO business_trips (commissioner_id, start_date, end_date) VALUES (:commissioner_id, :start_date, :end_date)";
        $stmt = $pdo->prepare($sql);
        $stmt->execute(['commissioner_id' => $commissioner_id, 'start_date' => $start_date, 'end_date' => $end_date]);
        echo "<script>window.location.href = '../home.php';</script>";
    } else {
        echo "<script>window.location.href = '../home.php';</script>";
    }
}
