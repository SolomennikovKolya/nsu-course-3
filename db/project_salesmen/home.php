<!-- Основная страница админа -->

<!DOCTYPE html>
<html lang="ru">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Коммивояжеры</title>
    <link rel="stylesheet" href="css/styles.css">
</head>

<body>
    <!-- Большой заголовок -->
    <header>
        <h1>Система для учета коммивояжеров</h1>
    </header>

    <section>
        <!-- Кнопка возвращения на index страницу -->
        <a href="index.php" class="home-button">Выйти</a><br><br>

        <!-- Кнопка для перехода на страницу лидеров по эффективности -->
        <h2>Топ сотрудников</h2>
        <button onclick="window.location.href='php/leaders.php'">Топ по эффективности</button><br><br>

        <!-- Формы для добавления коммивояжеров, товаров и командировок  -->
        <h2>Добавить новые данные</h2>
        <div class="form-container">
            <form action="php/add_commissioner.php" method="POST">
                <h3>Добавить коммивояжера</h3>
                <input type="text" name="full_name" placeholder="Ф.И.О." required>
                <input type="text" name="address" placeholder="Адрес">
                <input type="text" name="phone" placeholder="Телефон" required>
                <button type="submit">Добавить</button>
            </form>

            <form action="php/add_product.php" method="POST">
                <h3>Добавить товар</h3>
                <input type="text" name="name" placeholder="Название товара" required>
                <input type="number" step="0.01" name="price" placeholder="Цена" required>
                <select name="unit" required>
                    <option value="шт">Штука</option>
                    <option value="кг">Килограмм</option>
                </select>
                <button type="submit">Добавить</button>
            </form>

            <form action="php/add_trip.php" method="POST">
                <h3>Добавить командировку</h3>
                <label for="commissioner_id">Выберите коммивояжера:</label>
                <select name="commissioner_id" required>
                    <?php
                    require_once 'php/connection.php';
                    $stmt = $pdo->query("SELECT id, full_name FROM commissioners");
                    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
                        echo "<option value='{$row['id']}'>{$row['id']} - {$row['full_name']}</option>";
                    }
                    ?>
                </select>
                <input type="date" name="start_date" placeholder="Дата начала" required>
                <input type="date" name="end_date" placeholder="Дата окончания" required>
                <button type="submit">Добавить</button>
            </form>

        </div>
    </section>

    <!-- Секция данных -->
    <section>
        <h2>Данные</h2>
        <!-- Вкладки (табы) -->
        <div class="tabs">
            <button class="tab-button active" data-tab="commissioners">Коммивояжеры</button>
            <button class="tab-button" data-tab="products">Товары</button>
            <button class="tab-button" data-tab="trips">Командировки</button>
        </div>

        <!-- Коммивояжеры -->
        <div class="tab-content active" id="commissioners">
            <h3>Коммивояжеры</h3>
            <div class="filter-sort">
                <input type="text" id="commissioners-search" placeholder="Фильтр по Ф.И.О."
                    oninput="filterTable('commissioners')">
            </div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Ф.И.О.</th>
                        <th>Адрес</th>
                        <th>Телефон</th>
                    </tr>
                </thead>
                <tbody id="commissioners-table">
                    <?php
                    require_once 'php/connection.php';
                    $stmt = $pdo->query("SELECT * FROM commissioners");
                    // Печать по одной строке
                    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
                        echo "<tr data-phone='{$row['phone']}'>
                                <td>{$row['id']}</td>
                                <td>{$row['full_name']}</td>
                                <td>{$row['address']}</td>
                                <td>{$row['phone']}</td>
                              </tr>";
                    }
                    ?>
                </tbody>
            </table>
        </div>

        <!-- Товары -->
        <div class="tab-content" id="products">
            <h3>Товары</h3>
            <div class="filter-sort">
                <input type="text" id="products-search" placeholder="Фильтр по названию"
                    oninput="filterTable('products')">
                <select id="products-unit-filter" onchange="filterTable('products')">
                    <option value="">Все единицы</option>
                    <option value="шт">Штука</option>
                    <option value="кг">Килограмм</option>
                </select>
                <button onclick="sortTable('products', 2)">Сортировать по цене</button>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Название</th>
                        <th>Цена</th>
                        <th>Единица измерения</th>
                    </tr>
                </thead>
                <tbody id="products-table">
                    <?php
                    $stmt = $pdo->query("SELECT * FROM products");
                    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
                        echo "<tr data-unit='{$row['unit']}'>
                                <td>{$row['id']}</td>
                                <td>{$row['name']}</td>
                                <td>{$row['price']}</td>
                                <td>{$row['unit']}</td>
                              </tr>";
                    }
                    ?>
                </tbody>
            </table>
        </div>

        <!-- Модальное окно. Сюда джава скрипт будет писать инфу -->
        <div id="report-modal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <div id="modal-report-content">
                    <!-- Контент отчета будет загружен сюда -->
                </div>
            </div>
        </div>

        <!-- Таблица командировок с кнопкой "Смотреть отчет" -->
        <div class="tab-content" id="trips">
            <h3>Командировки</h3>
            <div class="filter-sort">
                <input type="text" id="trips-search" placeholder="Фильтр по Ф.И.О. коммивояжера"
                    oninput="filterTable('trips')">
                <input type="date" id="trips-start-date" placeholder="От даты" onchange="filterByDate()">
                <input type="date" id="trips-end-date" placeholder="До даты" onchange="filterByDate()">
            </div>

            <table id="trips-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Коммивояжер</th>
                        <th>Дата начала</th>
                        <th>Дата окончания</th>
                        <th>Действия</th>
                    </tr>
                </thead>
                <tbody>
                    <?php
                    require_once 'php/connection.php';
                    $stmt = $pdo->query("SELECT bt.id, bt.start_date, bt.end_date, c.full_name FROM business_trips bt JOIN commissioners c ON bt.commissioner_id = c.id");
                    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
                        echo "<tr>
                        <td>{$row['id']}</td>
                        <td>{$row['full_name']}</td>
                        <td>{$row['start_date']}</td>
                        <td>{$row['end_date']}</td>
                        <td><button class='report-button' data-id='{$row['id']}'>Смотреть отчет</button></td>
                      </tr>";
                    }
                    ?>
                </tbody>
            </table>
        </div>
    </section>

    <!-- Этот код устанавливает функциональность для работы с модальным окном и кнопками, которые показывают подробный отчет -->
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const buttons = document.querySelectorAll('.tab-button');
            const contents = document.querySelectorAll('.tab-content');

            buttons.forEach(button => {
                button.addEventListener('click', () => {
                    const tab = button.getAttribute('data-tab');
                    buttons.forEach(btn => btn.classList.remove('active'));
                    contents.forEach(content => content.classList.remove('active'));
                    button.classList.add('active');
                    document.getElementById(tab).classList.add('active');
                });
            });

            // Автозаполнение дат
            const today = new Date();
            const oneMonthLater = new Date(today);
            oneMonthLater.setMonth(today.getMonth() + 1);

            const startDateInput = document.querySelector('input[name="start_date"]');
            const endDateInput = document.querySelector('input[name="end_date"]');

            if (startDateInput && endDateInput) {
                startDateInput.value = today.toISOString().split('T')[0];
                endDateInput.value = oneMonthLater.toISOString().split('T')[0];
            }
        });

        document.addEventListener('DOMContentLoaded', () => {
            const modal = document.getElementById("report-modal");
            const closeModal = document.querySelector(".close");

            const buttons = document.querySelectorAll('.report-button');

            buttons.forEach(button => {
                button.addEventListener('click', () => {
                    const tripId = button.getAttribute('data-id');

                    // Загружаем отчет через AJAX
                    fetch(`php/full_report.php?trip_id=${tripId}`)
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

        // Фильтрация соответствующей таблицы (в зависимости от открытой вкладки). Скрывает ненужные элементы
        function filterTable(tab) {
            // Получаем текст из поля ввода поиска
            const searchValue = document.getElementById(`${tab}-search`).value.toLowerCase();
            // Получаем список строк таблицы
            const rows = document.querySelectorAll(`#${tab}-table tr`);

            if (tab === 'products') {
                const unitFilter = document.getElementById('products-unit-filter').value;
                rows.forEach(row => {
                    // Проверяем, содержит ли текст строки искомое значение
                    const nameMatch = row.innerText.toLowerCase().includes(searchValue);
                    // Проверяем, соответствует ли единица измерения строке в фильтре (или фильтр не указан)
                    const unitMatch = unitFilter === "" || row.getAttribute('data-unit') === unitFilter;
                    // Отображение или скрытие строки
                    row.style.display = nameMatch && unitMatch ? '' : 'none';
                });
            } else if (tab === 'trips') {
                rows.forEach(row => {
                    const text = row.children[1].innerText.toLowerCase();
                    row.style.display = text.includes(searchValue) ? '' : 'none';
                });
            } else {
                rows.forEach(row => {
                    const text = row.innerText.toLowerCase();
                    row.style.display = text.includes(searchValue) ? '' : 'none';
                });
            }
        }

        // Фильтрация командировок по дате
        function filterByDate() {
            const startDate = document.getElementById('trips-start-date').value;
            const endDate = document.getElementById('trips-end-date').value;
            const rows = document.querySelectorAll('#trips-table tr');

            rows.forEach(row => {
                const rowStartDate = row.cells[2].innerText; // Дата начала
                const rowEndDate = row.cells[3].innerText; // Дата окончания

                const rowStart = new Date(rowStartDate);
                const rowEnd = new Date(rowEndDate);

                let match = true;

                if (startDate) {
                    const filterStart = new Date(startDate);
                    if (rowStart < filterStart) match = false;
                }

                if (endDate) {
                    const filterEnd = new Date(endDate);
                    if (rowEnd > filterEnd) match = false;
                }

                row.style.display = match ? '' : 'none';
            });
        }

        // Сортировка товаров по цене
        function sortTable(tab, colIndex) {
            const table = document.querySelector(`#${tab}-table`);
            const rows = Array.from(table.querySelectorAll('tr'));

            rows.sort((a, b) => {
                const aText = a.children[colIndex].innerText;
                const bText = b.children[colIndex].innerText;
                return colIndex === 2 ? parseFloat(aText) - parseFloat(bText) : aText.localeCompare(bText, 'ru', {
                    numeric: true
                });
            });

            rows.forEach(row => table.appendChild(row));
        }
    </script>
</body>

</html>