
### Проблема конкурентного доступа к разделяемому ресурсу.
- В каталоге sync репозитория git@github.com:mrutman/os.git вы найдете простую реализацию очереди на списке. Изучите код, соберите и запустите программу queue-example.c. Посмотрите вывод программы и убедитесь что он соответствует вашему пониманию работы данной реализации очереди. Добавьте реализацию функции queue_destroy().
- Изучите код программы queue-threads.c и разберитесь что она делает. Соберите программу.
	- Запустите программу несколько раз. Если появляются ошибки выполнения, попытайтесь их объяснить и определить что именно вызывает ошибку. Какие именно ошибки вы наблюдали?
	- Поиграйте следующими параметрами:
		- Размером очереди (задается в queue_init()). Запустите программу с размером очереди от 1000 до 1000000.
		- Привязкой к процессору (задается функцией set_cpu()). Привяжите потоки к одному процессору (ядру) и к разным.
		- Планированием потоков (функция sched_yield()). Попробуйте убрать эту функцию перед созданием второго потока.
		- Объясните наблюдаемые результаты.