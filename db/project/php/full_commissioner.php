<?php
require_once 'connection.php';

// Извлекается параметр commissioner_id из URL (GET-запроса).
$commissioner_id = $_GET['commissioner_id'] ?? null;
if (!$commissioner_id) {
    die("Ошибка: Не указан ID коммивояжера.");
}

// Получение коммивояжера по его id из базы данных
$stmt = $pdo->prepare("SELECT * FROM commissioners WHERE id = :commissioner_id"); // Подготавливается запрос (commissioner_id - параметр)
$stmt->execute(['commissioner_id' => $commissioner_id]); // Выполнение запроса с подстановкой параметров
$commissioner = $stmt->fetch(PDO::FETCH_ASSOC);
if (!$commissioner) {
    die("Ошибка: Коммивояжер не найден.");
}

// Получение текущей командировки в зависимости от текущего времени
$current_date = date('Y-m-d');
$stmt = $pdo->prepare("SELECT * FROM business_trips WHERE commissioner_id = :commissioner_id AND :current_date BETWEEN start_date AND end_date ORDER BY start_date DESC LIMIT 1");
$stmt->execute(['commissioner_id' => $commissioner_id, 'current_date' => $current_date]);
$current_trip = $stmt->fetch(PDO::FETCH_ASSOC);

// Возвращает список товаров (id, name, unit), а именно массив ассоциативных массивов (массив словарей)
// COALESCE(unit, 'шт') - возвращает значение из столбца init, либо 'шт' если значение в столбце NULL
$products = $pdo->query("SELECT id, name, COALESCE(unit, 'шт') AS unit FROM products")->fetchAll(PDO::FETCH_ASSOC);

// Извлечение информации о товарах, связанных с текущей командировкой
// Здесь происходит соединение таблиц products_in_trip pit, products и products_returned
// pit — это алиас (сокращение) для таблицы products_in_trip, так же как p для products и pr для products_returned
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
?>

<!DOCTYPE html>
<html lang="ru">

<head>
    <meta charset="UTF-8">
    <title>Командировка - <?php echo htmlspecialchars($commissioner['full_name'], ENT_QUOTES, 'UTF-8'); ?></title>
    <!-- Подключает файл стилей (CSS) к HTML-документу -->
    <link rel="stylesheet" href="../css/commissioner_style.css">

    <script>
        // Фильтрует строки таблицы на основе введённого значения.
        function filterTable() {
            const searchValue = document.getElementById('product-search').value.toLowerCase();
            const rows = document.querySelectorAll('#products-table tr');
            rows.forEach(row => {
                const productName = row.cells[0].innerText.toLowerCase();
                const quantityTaken = row.cells[2].innerText.toLowerCase();
                const quantitySold = row.cells[3].innerText.toLowerCase();
                const remainingQuantity = row.cells[4].innerText.toLowerCase();
                const match = productName.includes(searchValue) ||
                    quantityTaken.includes(searchValue) ||
                    quantitySold.includes(searchValue) ||
                    remainingQuantity.includes(searchValue);
                row.style.display = match ? '' : 'none';
            });
        }

        // Сортирует строки таблицы по выбранному критерию.
        function sortTable() {
            const table = document.getElementById('products-table');
            const rows = Array.from(table.rows).slice(0); // Пропускаем заголовок
            const sortBy = document.getElementById('sort-by').value;

            rows.sort((a, b) => {
                let valueA, valueB;

                switch (sortBy) {
                    case 'price':
                        valueA = parseFloat(a.cells[1].innerText.replace(' ₽', '').replace(',', '.'));
                        valueB = parseFloat(b.cells[1].innerText.replace(' ₽', '').replace(',', '.'));
                        break;
                    case 'quantity_taken':
                        valueA = parseFloat(a.cells[2].innerText);
                        valueB = parseFloat(b.cells[2].innerText);
                        break;
                    case 'quantity_sold':
                        valueA = parseFloat(a.cells[3].innerText);
                        valueB = parseFloat(b.cells[3].innerText);
                        break;
                    case 'quantity_remaining':
                        valueA = parseFloat(a.cells[4].innerText);
                        valueB = parseFloat(b.cells[4].innerText);
                        break;
                    default:
                        return 0;
                }

                return valueA - valueB;
            });

            rows.forEach(row => table.appendChild(row)); // Перемещаем отсортированные строки
        }

        // Настраивает шаг ввода количества товара в зависимости от единицы измерения (шт или дробное значение).
        function checkUnit() {
            const select = document.getElementById('product-select');
            const selectedOption = select.options[select.selectedIndex];
            const unit = selectedOption.getAttribute('data-unit');
            const quantityInput = document.getElementById('quantity-taken');

            if (unit === 'шт') {
                quantityInput.step = '1'; // Ограничиваем только целые числа
            } else {
                quantityInput.step = '0.01'; // Разрешаем дробные значения
            }
        }

        // Проверяет корректность введённого количества перед отправкой формы.
        function validateQuantity() {
            const quantity = document.getElementById('quantity-taken').value;

            if (quantity < 0) {
                alert('Количество не может быть меньше нуля.');
                return false;
            }

            const select = document.getElementById('product-select');
            const unit = select.options[select.selectedIndex].getAttribute('data-unit');

            if (unit === 'шт' && !Number.isInteger(parseFloat(quantity))) {
                alert('Для товаров с единицей измерения "шт" допускаются только целые числа.');
                return false;
            }

            return true;
        }

        // Установить начальное ограничение при загрузке страницы
        document.addEventListener('DOMContentLoaded', checkUnit);
    </script>

</head>

<body>
    <section>
        <a href="../index.php" class="home-button">Выйти</a>
        <h2>Информация о коммивояжере</h2>
        <p><strong>Ф.И.О.: </strong><?php echo htmlspecialchars($commissioner['full_name'], ENT_QUOTES, 'UTF-8'); ?></p>
        <p><strong>Телефон: </strong><?php echo htmlspecialchars($commissioner['phone'], ENT_QUOTES, 'UTF-8'); ?></p>
        <a href="all_reports.php?commissioner_id=<?php echo $commissioner_id; ?>">
            <button>Смотреть все командировки</button>
        </a><br><br>

        <?php if ($current_trip): ?>
            <h2>Текущая командировка</h2>
            <p><strong>Дата начала:</strong> <?php echo htmlspecialchars($current_trip['start_date']); ?></p>
            <p><strong>Дата окончания:</strong> <?php echo htmlspecialchars($current_trip['end_date']); ?></p>

            <h3>Добавить товар</h3>
            <form method="POST" action="add_product_in_trip.php" onsubmit="return validateQuantity()">
                <input type="hidden" name="trip_id" value="<?php echo $current_trip['id']; ?>">
                <input type="hidden" name="commissioner_id" value="<?php echo $commissioner_id; ?>">
                <label>Товар:</label>
                <select id="product-select" name="product_id" onchange="checkUnit()" required>
                    <?php foreach ($products as $product): ?>
                        <option value="<?php echo $product['id']; ?>" data-unit="<?php echo $product['unit']; ?>">
                            <?php echo htmlspecialchars($product['name']); ?> (<?php echo $product['unit']; ?>)
                        </option>
                    <?php endforeach; ?>
                </select>
                <label>Количество:</label>
                <input type="number" id="quantity-taken" name="quantity_taken" step="1" min="0" required>
                <button type="submit">Добавить</button>
            </form>

            <h3>Товары в командировке</h3>
            <div class="filter-sort">
                <input type="text" id="product-search" placeholder="Фильтровать по названию" oninput="filterTable()">
                <select id="sort-by" onchange="sortTable()">
                    <option value="price">Цена</option>
                    <option value="quantity_taken">Взято</option>
                    <option value="quantity_sold">Продано</option>
                    <option value="quantity_remaining">Осталось</option>
                </select>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Товар</th>
                        <th>Цена</th>
                        <th>Взято</th>
                        <th>Продано</th>
                        <th>Осталось</th>
                        <th>Действие</th>
                    </tr>
                </thead>
                <tbody id="products-table">
                    <?php foreach ($current_trip_products as $product): ?>
                        <tr>
                            <td><?php echo htmlspecialchars($product['name']); ?></td>
                            <td><?php echo htmlspecialchars($product['price'], ENT_QUOTES, 'UTF-8'); ?> ₽</td>
                            <?php
                            $stmt = $pdo->prepare("
                                    SELECT quantity_returned
                                    FROM products_returned 
                                    WHERE trip_id = :trip_id AND product_id = :product_id
                                ");
                            $stmt->execute(['trip_id' => $current_trip['id'], 'product_id' => $product['product_id']]);
                            $returned = $stmt->fetch(PDO::FETCH_ASSOC);
                            $quantity_returned = $returned ? $returned['quantity_returned'] : 0;

                            // Рассчитываем количество проданных товаров
                            $quantity_sold = $product['quantity_taken'] - $quantity_returned;
                            ?>
                            <td id="quantity_taken_<?php echo $product['product_id']; ?>">
                                <?php echo $product['quantity_taken']; ?>
                            </td>
                            <td id="quantity_sold_<?php echo $product['product_id']; ?>"><?php echo $quantity_sold; ?></td>
                            <td id="quantity_remaining_<?php echo $product['product_id']; ?>">
                                <?php
                                $remaining = $product['quantity_taken'] - $quantity_sold;
                                echo $remaining;
                                ?>
                            </td>
                            <td>
                                <form method="POST" action="mark_sold.php">
                                    <input type="hidden" name="trip_id" value="<?php echo $current_trip['id']; ?>">
                                    <input type="hidden" name="product_id" value="<?php echo $product['product_id']; ?>">
                                    <input type="hidden" name="commissioner_id" value="<?php echo $commissioner_id; ?>">
                                    <input type="number" id="quantity_sold_<?php echo $product['product_id']; ?>"
                                        name="quantity_sold" step="0.01" placeholder="Продано" required>
                                    <input type="hidden" id="quantity_returned_<?php echo $product['product_id']; ?>"
                                        value="<?php echo $quantity_returned; ?>">
                                    <button type="submit"
                                        id="sell_button_<?php echo $product['product_id']; ?>">Продать</button>
                                </form>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        <?php else: ?>
            <p>Нет активной командировки.</p>
        <?php endif; ?>
    </section>
</body>

</html>