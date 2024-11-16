<!-- Указывает браузеру, что используется HTML5. Это декларация типа документа, необходимая для правильной интерпретации HTML-кода. -->
<!DOCTYPE html>
<!-- Открывает HTML-документ. Атрибут lang="ru" указывает, что язык документа русский, 
что помогает браузерам и поисковым системам адаптировать обработку текста (например, определить правильные правила орфографии). -->
<html lang="ru">

<!-- Открывает секцию метаданных документа. Здесь размещается информация, которая не отображается на странице напрямую (например, заголовок, кодировка). -->

<head>
    <!-- Указывает, что документ использует кодировку UTF-8. Это необходимо для корректного отображения символов, таких как кириллица. -->
    <meta charset="UTF-8">
    <!-- Определяет заголовок страницы, который отображается на вкладке браузера. -->
    <title>Подключение к базе данных</title>
</head>

<!-- Открывает тело документа. Все элементы, которые будут видны пользователю, размещаются внутри этого тега. -->

<body>
    <!-- Добавляет заголовок второго уровня. Этот заголовок информирует пользователя о назначении формы. -->
    <h2>Введите данные для подключения к базе данных</h2>
    <!-- Создаёт форму для отправки данных. action - указывает URL-адрес, куда будут отправлены данные формы
    method - задаёт метод передачи данных (POST). Этот метод передаёт данные в теле HTTP-запроса. -->
    <form action="php/create_connection.php" method="post">
        <!-- Поле для ввода имени пользователя -->
        <!-- Метка для текстового поля. Атрибут for связывает метку с элементом <input> по его id. -->
        <label for="username">Имя пользователя:</label>
        <!-- Текстовое поле. name - задаёт имя переменной, под которой это поле будет отправлено на сервер. 
        id - связывает поле с меткой. required - делает поле обязательным для заполнения. -->
        <input type="text" name="username" id="username" required><br><br>

        <!-- Поле для ввода пароля -->
        <label for="password">Пароль:</label>
        <input type="password" name="password" id="password"><br><br>

        <!-- Выпадающий список для выбора роли: -->
        <label for="role">Выберите роль:</label>
        <!-- <select> создаёт выпадающий список. -->
        <select name="role" id="role" required>
            <!-- <option>: задаёт варианты выбора. Атрибут value указывает значение, которое будет отправлено на сервер при выборе. -->
            <option value="admin">Администратор</option>
            <option value="commissioner">Коммивояжер</option>
        </select><br><br>

        <!-- Условное поле для ввода ID коммивояжер -->
        <!-- <div>: контейнер, который скрыт изначально (style="display: none;"). -->
        <div id="commissioner-id-container" style="display: none;">
            <label for="commissioner_id">Введите ID коммивояжера:</label>
            <input type="number" name="commissioner_id" id="commissioner_id"><br><br>
        </div>

        <!-- Кнопка, при нажатии на которую данные формы отправляются на сервер -->
        <button type="submit">Подключиться</button>
    </form>

    <!-- JavaScript секция для интерактивных действий -->
    <script>
        const roleSelect = document.getElementById('role');
        const commissionerIdContainer = document.getElementById('commissioner-id-container');

        // Вызывает функцию при изменении значения в списке.
        roleSelect.addEventListener('change', function () {
            if (roleSelect.value === 'commissioner') {
                commissionerIdContainer.style.display = 'block';
            } else {
                commissionerIdContainer.style.display = 'none';
            }
        });
    </script>
</body>

</html>