
### Места (Асинхронное сетевое взаимодействие)

Используя методы асинхронного программирования (например, CompletableFuture для Java) или библиотеки реактивного программирования (RxJava, например) провзаимодействовать с несколькими публично доступными API и сделать приложение с любым интерфейсом, основанное на этих API. При этом API должны использоваться так:
- Все вызовы должны делаться с помощью HTTP-библиотеки с асинхронных интерфейсом;
- Все независимые друг от друга вызовы API должны работать одновременно;
- Вызовы API, которые зависят от данных, полученных из предыдущих API, должны оформляться в виде асинхронной цепочки вызовов;
- Все результаты работы должны собираться в общий объект, поэтапный вывод результатов в процессе работы недопустим;
- Не допускаются блокировки на ожидании промежуточных результатов в цепочке вызовов, допустима только блокировка на ожидании конечного результата (в случае консольного приложения).
- Другими словами, логика программы должна быть оформлена как две функции, каждая из которых возвращает CompletableFuture (или аналог в вашем ЯП) без блокировок. Первая функция выполняет п.2, а вторая — п.п. 4 и 5 из списка ниже.

Логика работы:
1. В поле ввода пользователь вводит название чего-то (например "Цветной проезд") и нажимает кнопку поиска;
2. Ищутся варианты локаций с помощью метода [1] и показываются пользователю в виде списка;
3. Пользователь выбирает одну локацию;
4. С помощью метода [2] ищется погода в локации;
5. С помощью метода [3] ищутся интересные места в локации, далее для каждого найденного места с помощью метода [4] ищутся описания;
6. Всё найденное показывается пользователю.

Методы API:
- [1] [получение локаций с координатами и названиями](https://docs.graphhopper.com/#operation/getGeocode)
- [2] [получение погоды по координатам](https://openweathermap.org/current)
- [3] [получение списка интересных мест по координатам](https://dev.opentripmap.org/docs#/Objects%20list/getListOfPlacesByRadius)
- [4] [получение описания места по его id](https://dev.opentripmap.org/docs#/Object%20properties/getPlaceByXid)

Баллов за задачу: 3.
