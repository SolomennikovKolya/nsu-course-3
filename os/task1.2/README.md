
### Потоки Joinable and Detached.
- Напишите программу, в которой основной поток будет дожидаться завершения созданного потока.
- Измените программу так чтобы созданный поток возвращал число 42, а основной поток получал это число и распечатывал.
- Измените программу так чтобы созданный поток возвращал указатель на строку “hello world”, а основной поток получал этот указатель и распечатывал строку.
- Напишите программу, которая в бесконечном цикле будет создавать поток, с поточной функцией, которая выводит свой идентификатор потока и завершается. Запустите. Объясните результат.
- Добавьте вызов pthread_detach() в поточную функцию. Объясните результат.
- Вместо вызова pthread_detach() передайте в pthread_create() аргументы, задающие тип потока- DETACHED. Запустите, убедитесь что поведение не изменилось.
