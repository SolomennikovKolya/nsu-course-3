#include "queue.h"

#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define RED "\033[41m"
#define NOCOLOR "\033[0m"

// Параметры программы
int max_count, reader_core, writer_core;

// Привязка текущего потока к конкретному процессорному ядру
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

	// printf("set_cpu: set cpu %d\n", n);
}

// Поток читающий из очереди
void *reader(void *arg)
{
	int expected = 0;
	queue_t *q = (queue_t *)arg;
	printf("reader:\t[%d %d %d]\n", getpid(), getppid(), gettid());

	set_cpu(reader_core);

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

// Поток пишущий в очередь
void *writer(void *arg)
{
	int i = 0;
	queue_t *q = (queue_t *)arg;
	printf("writer:\t[%d %d %d]\n", getpid(), getppid(), gettid());

	set_cpu(writer_core);

	while (1)
	{
		int ok = queue_add(q, i);
		if (!ok)
			continue;
		i++;
		// usleep(1);
	}

	return NULL;
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "main: usage: ./prog.out <max_count> <reader_core> <writer_core>\n");
		return -1;
	}

	max_count = atoi(argv[1]);
	reader_core = atoi(argv[2]);
	writer_core = atoi(argv[3]);
	pthread_t tid;
	queue_t *q;
	int err;

	printf("main:\t[%d %d %d]\n", getpid(), getppid(), gettid());

	q = queue_init(max_count);

	err = pthread_create(&tid, NULL, reader, q);
	if (err)
	{
		fprintf(stderr, "main: pthread_create() failed: %s\n", strerror(err));
		return -1;
	}

	// sched_yield();

	err = pthread_create(&tid, NULL, writer, q);
	if (err)
	{
		fprintf(stderr, "main: pthread_create() failed: %s\n", strerror(err));
		return -1;
	}

	pthread_exit(NULL);
	return 0;
}
