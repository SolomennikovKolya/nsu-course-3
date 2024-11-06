#include "mythread/mythread.h"
#include <stdio.h>
#include <unistd.h>

void *test(void *arg)
{
	printf("Thread: Hello\n");
	// printf("Thread: %d\n", mythread_self());

	return NULL;
}

int main()
{
	printf("\n");
	printf("Main: %d\n", getpid());
	// printf("Main: %d\n", mythread_self());

#if 1
	mythread_t t1, t2, t3;
	mythread_create(&t1, test, NULL);
	mythread_create(&t2, test, NULL);
	mythread_create(&t3, test, NULL);
#else
	mythread_t thread;
	if (mythread_create(&thread, test, NULL) != 0)
	{
		fprintf(stderr, "Ошибка при создании потока\n");
		return 1;
	}
#endif

	sleep(1);
	return 0;
}
