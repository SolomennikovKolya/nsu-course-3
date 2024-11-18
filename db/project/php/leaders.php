<?php
require_once 'connection.php';

$stmt = $pdo->query("
    SELECT 
        c.full_name, 
        COUNT(DISTINCT bt.id) AS trips_count,
        SUM((pit.quantity_taken - COALESCE(pr.quantity_returned, 0)) * p.price) AS revenue, 
        SUM(pit.quantity_taken * p.price) AS total_taken_price
    FROM commissioners c
    JOIN business_trips bt ON c.id = bt.commissioner_id
    JOIN products_in_trip pit ON bt.id = pit.trip_id
    LEFT JOIN products_returned pr ON pit.trip_id = pr.trip_id AND pit.product_id = pr.product_id
    JOIN products p ON pit.product_id = p.id
    GROUP BY c.id
    HAVING total_taken_price > 0
    ORDER BY revenue / total_taken_price DESC
");

$commissioners = $stmt->fetchAll(PDO::FETCH_ASSOC);
?>

<!DOCTYPE html>
<html lang="ru">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Топ сотрудников</title>
    <link rel="stylesheet" href="../css/styles.css">
</head>

<body>
    <header>
        <h1>Топ сотрудников по эффективности</h1>
    </header>
    <section>
        <a href="../home.php" class="home-button">Домой</a>
        <input type="text" id="search" placeholder="Фильтр по имени" oninput="filterTable()">
        <label for="sort-order">Сортировать по:</label>
        <select id="sort-column" onchange="sortTable()">
            <option value="efficiency">Эффективность</option>
            <option value="revenue">Выручка</option>
            <option value="trips">Кол-во командировок</option>
        </select>
        <select id="sort-order" onchange="sortTable()">
            <option value="desc">Убыванию</option>
            <option value="asc">Возрастанию</option>
        </select>
        <table id="leaders-table">
            <thead>
                <tr>
                    <th>Ф.И.О.</th>
                    <th>Выручка (руб.)</th>
                    <th>Кол-во командировок</th>
                    <th>Эффективность (%)</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($commissioners as $commissioner):
                    $efficiency = $commissioner['revenue'] / $commissioner['total_taken_price'] * 100;
                ?>
                    <tr>
                        <td><?= htmlspecialchars($commissioner['full_name']) ?></td>
                        <td><?= number_format($commissioner['revenue'], 2, '.', '') ?></td>
                        <td><?= htmlspecialchars($commissioner['trips_count']) ?></td>
                        <td><?= number_format($efficiency, 2, '.', '') ?></td>
                    </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
    </section>
    <script>
        function filterTable() {
            const searchValue = document.getElementById('search').value.toLowerCase();
            const rows = document.querySelectorAll('#leaders-table tbody tr');
            rows.forEach(row => {
                const name = row.children[0].innerText.toLowerCase();
                row.style.display = name.includes(searchValue) ? '' : 'none';
            });
        }

        function sortTable() {
            const rows = Array.from(document.querySelectorAll('#leaders-table tbody tr'));
            const sortColumn = document.getElementById('sort-column').value;
            const sortOrder = document.getElementById('sort-order').value;

            rows.sort((a, b) => {
                let aValue, bValue;
                if (sortColumn === 'efficiency') {
                    aValue = parseFloat(a.children[3].innerText);
                    bValue = parseFloat(b.children[3].innerText);
                } else if (sortColumn === 'revenue') {
                    aValue = parseFloat(a.children[1].innerText);
                    bValue = parseFloat(b.children[1].innerText);
                } else if (sortColumn === 'trips') {
                    aValue = parseInt(a.children[2].innerText);
                    bValue = parseInt(b.children[2].innerText);
                }
                return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
            });

            const tbody = document.querySelector('#leaders-table tbody');
            rows.forEach(row => tbody.appendChild(row));
        }
    </script>
</body>

</html>