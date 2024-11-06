#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>

#include "queue.h"

#define RED "\033[41m"
#define NOCOLOR "\033[0m"

// Привязка потока к конкретному процессорному ядру
void set_cpu(int n)
{
	int err;
	cpu_set_t cpuset; // Это тип данных, который представляет набор процессорных ядер (CPU set)
	pthread_t tid = pthread_self();

	CPU_ZERO(&cpuset);	 // Очищает набор процессорных ядер
	CPU_SET(n, &cpuset); // Добавляет процессорное ядро с индексом n в набор cpuset

	// Установки привязки потока (tid) к набору процессорных ядер (cpuset)
	err = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
	if (err)
	{
		printf("set_cpu: pthread_setaffinity failed for cpu %d\n", n);
		return;
	}

	printf("set_cpu: set cpu %d\n", n);
}

void *reader(void *arg)
{
	int expected = 0;
	queue_t *q = (queue_t *)arg;
	printf("reader [%d %d %d]\n", getpid(), getppid(), gettid());

	set_cpu(1);

	while (1)
	{
		int val = -1;
		int ok = queue_get(q, &val);
		if (!ok)
			continue;

		if (expected != val)
		{
			printf(RED "ERROR: get value is %d but expected - %d" NOCOLOR "\n", val, expected);
			queue_print_stats(q);
		}

		expected = val + 1;
	}

	return NULL;
}

void *writer(void *arg)
{
	int i = 0;
	queue_t *q = (queue_t *)arg;
	printf("writer [%d %d %d]\n", getpid(), getppid(), gettid());

	set_cpu(1);

	while (1)
	{
		int ok = queue_add(q, i);
		if (!ok)
			continue;
		i++;
	}

	return NULL;
}

int main()
{
	pthread_t tid;
	queue_t *q;
	int err;

	printf("main [%d %d %d]\n", getpid(), getppid(), gettid());

	q = queue_init(1000000);

	err = pthread_create(&tid, NULL, reader, q);
	if (err)
	{
		printf("main: pthread_create() failed: %s\n", strerror(err));
		return -1;
	}

	sched_yield(); // Поток добровольно уступает оставшееся время своего кванта, давая другим потокам возможность быть выполненными.

	err = pthread_create(&tid, NULL, writer, q);
	if (err)
	{
		printf("main: pthread_create() failed: %s\n", strerror(err));
		return -1;
	}

	// TODO: join threads

	pthread_exit(NULL);

	return 0;
}
