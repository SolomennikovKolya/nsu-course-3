#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void *thread1_func(void *arg)
{
	printf("Thread 1: Блокирует все сигналы\n");

	// Блокируем все сигналы
	sigset_t set;
	sigfillset(&set);
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	while (1)
	{
		sleep(1);
	}
	return NULL;
}

void sigint_handler(int signo)
{
	printf("Thread 2: Получен сигнал SIGINT(%d)\n", signo);
}

void *thread2_func(void *arg)
{
	printf("Thread 2: Ждёт SIGINT\n");

	// Блокируем все сигналы кроме SIGINT
	sigset_t set;
	sigfillset(&set);
	sigdelset(&set, SIGINT);
	pthread_sigmask(SIG_SETMASK, &set, NULL);

	// Устанавливаем обработчик сигнала SIGINT
	struct sigaction sa;
	sa.sa_handler = sigint_handler;
	sa.sa_flags = SA_RESTART; // Автоматически перезапускает прерванные системные вызовы.
	sigfillset(&sa.sa_mask);  // Блокирует все сигналы, при выполнении обработчика

	if (sigaction(SIGINT, &sa, NULL) == -1)
	{
		perror("sigaction");
		exit(1);
	}

	while (1)
	{
		sleep(1);
	}
	return NULL;
}

void *thread3_func(void *arg)
{
	printf("Thread 3: Ждёт SIGQUIT\n");

	// Блокируем все сигналы кроме SIGQUIT
	sigset_t set;
	sigfillset(&set);
	sigdelset(&set, SIGQUIT);
	pthread_sigmask(SIG_SETMASK, &set, NULL);

	sigset_t wait_set;
	sigemptyset(&wait_set);
	sigaddset(&wait_set, SIGQUIT);
	int sig;

	while (1)
	{
		if (sigwait(&wait_set, &sig))
		{
			perror("sigwait");
			exit(1);
		}
		printf("Thread 3: Получен сигнал SIGQUIT(%d)\n", sig);
	}
	return NULL;
}

int main()
{
	printf("Main: %d\n", getpid());

	// Блокируем все сигналы
	sigset_t set;
	sigfillset(&set);
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	// Создаем потоки
	pthread_t thread1, thread2, thread3;
	pthread_create(&thread1, NULL, thread1_func, NULL);
	pthread_create(&thread2, NULL, thread2_func, NULL);
	pthread_create(&thread3, NULL, thread3_func, NULL);

	// Ожидаем завершения потоков
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);

	return 0;
}

/*
int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);
signum — номер сигнала, для которого устанавливается обработчик.
act    — указатель на структуру sigaction, которая определяет новый обработчик сигнала.
oldact — если не NULL, возвращает предыдущее состояние обработчика сигнала.

struct sigaction {
	void (*sa_handler)(int);    // Указатель на функцию-обработчик сигнала. Может быть SIG_IGN — игнорировать сигнал, SIG_DFL — выполнить действие по умолчанию.
	void (*sa_sigaction)(int, siginfo_t *, void *);  // Альтернативный обработчик
	sigset_t sa_mask;           // Набор сигналов, которые будут блокированы во время выполнения текущего обработчика. Это предотвращает прерывание обработчика другими сигналами.
	int sa_flags;               // Флаги, определяющие поведение обработчика
};

int sigwait(const sigset_t *set, int *sig);
set - указатель на набор сигналов (sigset_t), в котором перечислены сигналы, которые нужно ожидать.
sig - указатель на переменную, в которую sigwait поместит номер принятого сигнала после его доставки.
*/

/*
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

void signal_handler(int signo)
{
	printf("Thread: %ld - получен сигнал %d\n", syscall(SYS_gettid), signo);
	pthread_exit(NULL);
}

void *func1(void *arg)
{
	printf("Thread: %ld - запущен\n", syscall(SYS_gettid));

	signal(SIGUSR1, signal_handler); // Установка обработчика сигнала

	sigset_t set;							// Создаем пустой набор сигналов
	sigfillset(&set);						// Добавляет все возможные сигналы в набор сигналов set.
	pthread_sigmask(SIG_BLOCK, &set, NULL); // Устанавливаем маску для блокировки всех сигналов в этом потоке

	sleep(3);

	printf("Thread: %ld - завершён\n", syscall(SYS_gettid));
	return NULL;
}

int main()
{
	printf("Main: %ld - запущен\n", syscall(SYS_gettid));

	pthread_t t1;
	if (pthread_create(&t1, NULL, func1, NULL))
	{
		perror("Ошибка создания потока");
		return 1;
	}

	sleep(0.1); // Небольшая задержка, чтобы убедиться, что поток готов к приему сигналов

	// Отправка сигнала потоку
	if (pthread_kill(t1, SIGUSR1))
	{
		perror("Ошибка отправки сигнала");
		return 1;
	}

	pthread_join(t1, NULL);

	printf("Main: %ld - завершён\n", syscall(SYS_gettid));
	return 0;
}
*/
