#include "mythread/mythread.h"
#include <stdio.h>
#include <unistd.h>

void *test(void *arg)
{
	printf("Thread: %d\n", getpid());
	return NULL;
}

int main()
{
	printf("Main: %d\n", getpid());

	// mythread_t thread1;
	// mythread_create(&thread1, test, NULL);

	// mythread_t thread2;
	// mythread_create(&thread2, test, NULL);

	mythread_t thread;
	if (mythread_create(&thread, test, NULL) != 0)
	{
		fprintf(stderr, "Ошибка при создании потока\n");
		return 1;
	}

	sleep(1);
	return 0;
}
