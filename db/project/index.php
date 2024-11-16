<!DOCTYPE html>
<html lang="ru">

<head>
    <meta charset="UTF-8">
    <title>Подключение к базе данных</title>
</head>

<body>
    <h2>Введите данные для подключения к базе данных</h2>
    <form action="php/create_connection.php" method="post">
        <label for="username">Имя пользователя:</label>
        <input type="text" name="username" id="username" required><br><br>

        <label for="password">Пароль:</label>
        <input type="password" name="password" id="password"><br><br>

        <label for="role">Выберите роль:</label>
        <select name="role" id="role" required>
            <option value="admin">Администратор</option>
            <option value="commissioner">Коммивояжер</option>
        </select><br><br>

        <div id="commissioner-id-container" style="display: none;">
            <label for="commissioner_id">Введите ID коммивояжера:</label>
            <input type="number" name="commissioner_id" id="commissioner_id"><br><br>
        </div>

        <button type="submit">Подключиться</button>
    </form>

    <script>
        const roleSelect = document.getElementById('role');
        const commissionerIdContainer = document.getElementById('commissioner-id-container');

        roleSelect.addEventListener('change', function() {
            if (roleSelect.value === 'commissioner') {
                commissionerIdContainer.style.display = 'block';
            } else {
                commissionerIdContainer.style.display = 'none';
            }
        });
    </script>
</body>

</html>