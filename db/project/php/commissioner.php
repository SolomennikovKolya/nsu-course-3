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

$stmt = $pdo->prepare("SELECT bt.id, bt.start_date, bt.end_date FROM business_trips bt WHERE bt.commissioner_id = :commissioner_id");
$stmt->execute(['commissioner_id' => $commissioner_id]);
$trips = $stmt->fetchAll(PDO::FETCH_ASSOC);

$total_revenue = 0;
$total_earned = 0;
$total_taken = 0;
$total_returned = 0;
$percentage = 0.10;

foreach ($trips as $trip) {
    $stmt = $pdo->prepare("SELECT pit.quantity_taken, pr.quantity_returned, p.price
                           FROM products_in_trip pit
                           LEFT JOIN products p ON pit.product_id = p.id
                           LEFT JOIN products_returned pr ON pr.trip_id = pit.trip_id AND pr.product_id = pit.product_id
                           WHERE pit.trip_id = :trip_id");
    $stmt->execute(['trip_id' => $trip['id']]);
    $products = $stmt->fetchAll(PDO::FETCH_ASSOC);

    foreach ($products as $product) {
        $quantity_taken = $product['quantity_taken'];
        $quantity_returned = $product['quantity_returned'] ?? 0;
        $price = $product['price'];

        $revenue = $quantity_taken * $price;
        $total_revenue += $revenue;

        $total_earned += $revenue * $percentage;

        $total_taken += $quantity_taken;
        $total_returned += $quantity_returned;
    }
}

$efficiency = $total_returned > 0 ? ($total_taken / $total_returned) : 0;
?>

<!DOCTYPE html>
<html lang="ru">

<head>
    <meta charset="UTF-8">
    <title>Коммивояжер - <?php echo htmlspecialchars($commissioner['full_name'], ENT_QUOTES, 'UTF-8'); ?></title>
    <link rel="stylesheet" href="../css/styles.css">
</head>

<body>
    <section>
        <h1>Информация о коммивояжере</h1>
        <p><strong>Ф.И.О.: </strong><?php echo htmlspecialchars($commissioner['full_name'], ENT_QUOTES, 'UTF-8'); ?></p>
        <p><strong>Адрес: </strong><?php echo htmlspecialchars($commissioner['address'], ENT_QUOTES, 'UTF-8'); ?></p>
        <p><strong>Телефон: </strong><?php echo htmlspecialchars($commissioner['phone'], ENT_QUOTES, 'UTF-8'); ?></p>

        <h3>Расчет зарплаты и эффективности</h3>
        <p><strong>Зарплата за указанный период: </strong><?php echo number_format($total_earned, 2); ?> руб.</p>
        <p><strong>Эффективность работы: </strong><?php echo number_format($efficiency, 2); ?>%</p>

        <h3>Командировки</h3>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Дата начала</th>
                    <th>Дата окончания</th>
                    <th>Действия</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($trips as $trip): ?>
                    <tr>
                        <td><?php echo htmlspecialchars($trip['id'], ENT_QUOTES, 'UTF-8'); ?></td>
                        <td><?php echo htmlspecialchars($trip['start_date'], ENT_QUOTES, 'UTF-8'); ?></td>
                        <td><?php echo htmlspecialchars($trip['end_date'], ENT_QUOTES, 'UTF-8'); ?></td>
                        <td><button class="report-button" data-id="<?php echo $trip['id']; ?>">Смотреть отчет</button></td>
                    </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
    </section>

    <!-- Модальное окно для отчетов -->
    <div id="report-modal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <div id="modal-report-content">
                <!-- Контент отчета будет загружен сюда -->
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Открытие модального окна
            const modal = document.getElementById("report-modal");
            const closeModal = document.querySelector(".close");

            const buttons = document.querySelectorAll('.report-button');

            buttons.forEach(button => {
                button.addEventListener('click', () => {
                    const tripId = button.getAttribute('data-id');

                    fetch(`report.php?trip_id=${tripId}`)
                        .then(response => response.text())
                        .then(data => {
                            document.getElementById("modal-report-content").innerHTML = data;
                            modal.style.display = "block"; // Показываем модальное окно
                        });
                });
            });
            closeModal.addEventListener('click', () => {
                modal.style.display = "none";
            });

            window.addEventListener('click', (event) => {
                if (event.target === modal) {
                    modal.style.display = "none";
                }
            });
        });
    </script>
</body>

</html>