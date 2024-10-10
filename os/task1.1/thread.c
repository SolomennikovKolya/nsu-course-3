#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

const int THREADS_COUNT = 3;
int global_var = 4;

void *mythread(void *arg)
{
	int local_var = 1;
	static int local_static_var = 2;
	const int local_const_var = 3;

	printf("mythread [%d %d %d]: Hello from mythread!\npthread_self: %lu\nlocal: %p\tstatic: %p\tconst: %p\tglobal: %p\nlocal: %d\t\tstatic: %d\t\tconst: %d\t\tglobal: %d\n\n",
		   getpid(), getppid(), gettid(), pthread_self(), &local_var, &local_static_var, &local_const_var, &global_var, local_var, local_static_var, local_const_var, global_var);

	local_var++;
	global_var++;

	sleep(100);

	return NULL;
}

int main()
{
	pthread_t tids[THREADS_COUNT];
	int err;

	printf("main [%d %d %d]: Hello from main!\n\n", getpid(), getppid(), gettid());

	for (int i = 0; i < THREADS_COUNT; ++i)
	{
		err = pthread_create(tids + i, NULL, mythread, NULL);
		if (err)
		{
			printf("main: pthread_create() failed: %s\n", strerror(err));
			return -1;
		}
	}

	for (int i = 0; i < THREADS_COUNT; ++i)
	{
		err = pthread_join(tids[i], NULL);
		if (err)
		{
			printf("main: join() failed: %s\n", strerror(err));
			return -1;
		}
	}

	for (int i = 0; i < THREADS_COUNT; ++i)
		printf("tids[%d] = %lu\n", i, tids[i]);
	printf("\n");

	// pthread_exit(NULL);
	return 0;
}
