#include "mythread.h"
#include <stdio.h>
#include <syscall.h>
#include <unistd.h>

void *test_routine(void *)
{
	printf("Thread:\t[%d, %ld]\n", getpid(), syscall(SYS_gettid));
}

int main()
{
	printf("Main:\t[%d, %ld]\n", getpid(), syscall(SYS_gettid));

	mythread_t thread;
	if (mythread_create(&thread, test_routine, NULL))
	{
		fprintf(stderr, "Ошибка при создании потока\n");
		return 1;
	}

	sleep(1);

	if (mythread_join(thread, NULL))
	{
		fprintf(stderr, "Ошибка при присоединении потока\n");
		return 1;
	}

	// mythread_t t1, t2, t3;
	// mythread_create(&t1, test_routine, NULL);
	// mythread_create(&t2, test_routine, NULL);
	// mythread_create(&t3, test_routine, NULL);

	// mythread_join(t1, NULL);
	// mythread_join(t2, NULL);
	// mythread_join(t3, NULL);

	return 0;
}
