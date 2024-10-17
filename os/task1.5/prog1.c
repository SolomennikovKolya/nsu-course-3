// #include <bits/types/sigset_t.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// #define SIG_BLOCK 0

// Обработчик для SIGINT
void sigint_handler(int signo)
{
	printf("Thread 2: Caught SIGINT signal (Ctrl+C)\n");
}

// Первый поток: блокирует все сигналы
void *thread1_func(void *arg)
{
	printf("Thread 1: Blocking all signals.\n");

	sigset_t set;
	sigfillset(&set); // Блокируем все сигналы
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	// Бесконечный цикл, эмулирующий выполнение потока
	while (1)
	{
		sleep(1);
		printf("Thread 1: Running...\n");
	}
	return NULL;
}

// Второй поток: обрабатывает SIGINT (Ctrl+C)
void *thread2_func(void *arg)
{
	printf("Thread 2: Waiting for SIGINT (Ctrl+C).\n");

	// Устанавливаем обработчик сигнала SIGINT
	struct sigaction sa;
	sa.sa_handler = sigint_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;

	if (sigaction(SIGINT, &sa, NULL) == -1)
	{
		perror("sigaction");
		exit(1);
	}

	// Бесконечный цикл для ожидания сигнала
	while (1)
	{
		sleep(1);
		printf("Thread 2: Running...\n");
	}
	return NULL;
}

// Третий поток: ждет SIGQUIT через sigwait()
void *thread3_func(void *arg)
{
	printf("Thread 3: Waiting for SIGQUIT (Ctrl+\\).\n");

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGQUIT); // Добавляем SIGQUIT в набор

	int sig;
	// Ожидание сигнала SIGQUIT через sigwait()
	if (sigwait(&set, &sig) == 0)
	{
		if (sig == SIGQUIT)
		{
			printf("Thread 3: Caught SIGQUIT signal (Ctrl+\\)\n");
		}
	}
	else
	{
		perror("sigwait");
		exit(1);
	}

	return NULL;
}

int main()
{
	pthread_t thread1, thread2, thread3;
	sigset_t set;

	// Блокируем SIGINT и SIGQUIT для главного потока, чтобы эти сигналы обрабатывали потоки
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	sigaddset(&set, SIGQUIT);
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	// Создаем потоки
	pthread_create(&thread1, NULL, thread1_func, NULL);
	pthread_create(&thread2, NULL, thread2_func, NULL);
	pthread_create(&thread3, NULL, thread3_func, NULL);

	// Ожидаем завершения потоков (по факту программа будет бесконечной)
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);

	return 0;
}
