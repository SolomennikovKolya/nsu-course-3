<?php
// Получаем ID коммивояжера из URL
$commissioner_id = isset($_GET['commissioner_id']) ? (int)$_GET['commissioner_id'] : 0;

// Проверяем, что ID коммивояжера корректен
if ($commissioner_id > 0) {
    // Подключаемся к базе данных и выбираем командировки для указанного коммивояжера
    require_once 'connection.php';

    $stmt = $pdo->prepare("SELECT bt.id, bt.start_date, bt.end_date, c.full_name FROM business_trips bt JOIN commissioners c ON bt.commissioner_id = c.id WHERE bt.commissioner_id = :commissioner_id");
    $stmt->execute(['commissioner_id' => $commissioner_id]);
    $trips = $stmt->fetchAll(PDO::FETCH_ASSOC);
}
?>

<?php
require_once 'get_commissioner_data.php';
?>

<!DOCTYPE html>
<html lang="ru">

<head>
    <meta charset="UTF-8">
    <title>Командировки коммивояжера</title>
    <link rel="stylesheet" href="../css/commissioner_style.css">
    <script>
        function openReportModal(tripId) {
            const reportContainer = document.getElementById('tripReportContainer');
            reportContainer.innerHTML = 'Загрузка отчета...';

            fetch(`report.php?trip_id=${tripId}`)
                .then(response => response.text())
                .then(html => {
                    reportContainer.innerHTML = html;
                })
                .catch(error => {
                    reportContainer.innerHTML = 'Ошибка при загрузке отчета.';
                    console.error('Error:', error);
                });

            document.getElementById('reportModal').style.display = 'block';
        }

        function closeReportModal() {
            document.getElementById('reportModal').style.display = 'none';
        }

        // Фильтрация по дате
        function filterTable() {
            const startDateFilter = document.getElementById('start-date-filter').value;
            const endDateFilter = document.getElementById('end-date-filter').value;
            const rows = document.querySelectorAll('tbody tr');

            rows.forEach(row => {
                const startDate = row.cells[1].innerText;
                const endDate = row.cells[2].innerText;
                let isVisible = true;

                // Фильтрация по дате начала
                if (startDateFilter && new Date(startDate) < new Date(startDateFilter)) {
                    isVisible = false;
                }

                // Фильтрация по дате окончания
                if (endDateFilter && new Date(endDate) > new Date(endDateFilter)) {
                    isVisible = false;
                }

                row.style.display = isVisible ? '' : 'none';
            });
        }

        // Сортировка по дате
        function sortTable() {
            const table = document.querySelector('table');
            const rows = Array.from(table.rows).slice(1); // Пропускаем заголовок
            const sortBy = document.getElementById('sort-by').value;

            let sortFunc;

            if (sortBy === 'start_date_desc') {
                // Сортируем по дате начала, от самых новых (поздних)
                sortFunc = (a, b) => new Date(b.cells[1].innerText) - new Date(a.cells[1].innerText);
            } else if (sortBy === 'start_date_asc') {
                // Сортируем по дате начала, от самых старых (ранних)
                sortFunc = (a, b) => new Date(a.cells[1].innerText) - new Date(b.cells[1].innerText);
            }

            rows.sort(sortFunc);

            // Перемещаем отсортированные строки
            rows.forEach(row => table.appendChild(row));
        }
    </script>
</head>

<body>
    <a href="full_commissioner.php?commissioner_id=<?php echo $commissioner_id; ?>" class="home-button">Домой</a>
    <h2>Все командировки коммивояжера</h2>

    <div class="filter-sort">
        <label for="start-date-filter">Фильтровать по дате начала:</label>
        <input type="date" id="start-date-filter" name="start-date-filter" onchange="filterTable()">
        <label for="end-date-filter">Фильтровать по дате окончания:</label>
        <input type="date" id="end-date-filter" name="end-date-filter" onchange="filterTable()">

        <label for="sort-by">Сортировать по:</label>
        <select id="sort-by" onchange="sortTable()">
            <option value="start_date_desc">Самые новые</option>
            <option value="start_date_asc">Самые старые</option>
        </select>
    </div>

    <table>
        <thead>
            <tr>
                <th>Командировка ID</th>
                <th>Дата начала</th>
                <th>Дата окончания</th>
                <th>Коммивояжер</th>
                <th>Действие</th>
            </tr>
        </thead>
        <tbody>
            <?php foreach ($trips as $trip): ?>
                <tr>
                    <td><?php echo htmlspecialchars($trip['id']); ?></td>
                    <td><?php echo htmlspecialchars($trip['start_date']); ?></td>
                    <td><?php echo htmlspecialchars($trip['end_date']); ?></td>
                    <td><?php echo htmlspecialchars($trip['full_name']); ?></td>
                    <td><button onclick="openReportModal(<?php echo $trip['id']; ?>)">Смотреть отчет</button></td>
                </tr>
            <?php endforeach; ?>
        </tbody>
    </table>

    <div id="reportModal" style="display: none;">
        <div class="modal-content">
            <span onclick="closeReportModal()" class="close-button">&times;</span>
            <div id="tripReportContainer"></div>
        </div>
    </div>
</body>

</html>