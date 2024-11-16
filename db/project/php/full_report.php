<?php
require_once 'connection.php';

$trip_id = $_GET['trip_id'];

$stmt = $pdo->prepare("SELECT bt.id, bt.start_date, bt.end_date, c.full_name FROM business_trips bt JOIN commissioners c ON bt.commissioner_id = c.id WHERE bt.id = :trip_id");
$stmt->execute(['trip_id' => $trip_id]);
$trip = $stmt->fetch(PDO::FETCH_ASSOC);

$stmt = $pdo->prepare("SELECT p.name, pit.quantity_taken, pr.quantity_returned, p.price
                       FROM products_in_trip pit
                       LEFT JOIN products p ON pit.product_id = p.id
                       LEFT JOIN products_returned pr ON pr.trip_id = pit.trip_id AND pr.product_id = pit.product_id
                       WHERE pit.trip_id = :trip_id");
$stmt->execute(['trip_id' => $trip_id]);
$products = $stmt->fetchAll(PDO::FETCH_ASSOC);

$total_revenue = 0;
$total_earned = 0;
$total_taken_price = 0;

foreach ($products as $product) {
    $quantity_taken = $product['quantity_taken'];
    $quantity_returned = $product['quantity_returned'] ?? 0;
    $price_per_unit = $product['price'];

    $revenue = ($quantity_taken - $quantity_returned) * $price_per_unit;

    $price_of_taken = $quantity_taken * $price_per_unit;

    $total_revenue += $revenue;
    $total_earned += $revenue * 0.30;

    $total_taken_price += $price_of_taken;
}

if ($total_taken_price > 0) {
    $efficiency = ($total_revenue / $total_taken_price) * 100;
} else {
    $efficiency = 100;
}


?>

<h2>Отчет по командировке #<?php echo htmlspecialchars($trip['id'], ENT_QUOTES, 'UTF-8'); ?></h2>
<h4>Коммивояжер: <?php echo htmlspecialchars($trip['full_name'], ENT_QUOTES, 'UTF-8'); ?></h4>
<p>Дата начала: <?php echo htmlspecialchars($trip['start_date'], ENT_QUOTES, 'UTF-8'); ?></p>
<p>Дата окончания: <?php echo htmlspecialchars($trip['end_date'], ENT_QUOTES, 'UTF-8'); ?></p>

<p>Выручка командировки: <?php echo number_format($total_revenue, 2); ?> руб.</p>
<p>Заработок коммивояжера: <?php echo number_format($total_earned, 2); ?> руб.</p>
<p>Эффективность: <?php echo number_format($efficiency, 2); ?>%</p>

<h4>Товары в командировке</h4>
<table>
    <thead>
        <tr>
            <th>Товар</th>
            <th>Взято</th>
            <th>Возвращено</th>
            <th>Цена</th> <!-- Добавляем столбец "Цена" -->
        </tr>
    </thead>
    <tbody>
        <?php foreach ($products as $product): ?>
            <tr>
                <td><?php echo htmlspecialchars($product['name'], ENT_QUOTES, 'UTF-8'); ?></td>
                <td><?php echo htmlspecialchars($product['quantity_taken'], ENT_QUOTES, 'UTF-8'); ?></td>
                <td><?php echo isset($product['quantity_returned']) ? htmlspecialchars($product['quantity_returned'], ENT_QUOTES, 'UTF-8') : "Еще не закончилась"; ?></td>
                <td><?php echo number_format($product['price'], 2, '.', ''); ?> руб.</td> <!-- Выводим цену -->
            </tr>
        <?php endforeach; ?>
    </tbody>
</table>