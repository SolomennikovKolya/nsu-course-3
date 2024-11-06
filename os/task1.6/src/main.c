#include "mythread.h"
#include <stdio.h>
#include <unistd.h>

void *test_routine(void *)
{
	printf("Thread: Hello\n");
}

int main()
{
	printf("Main: Hello\n");

	mythread_t t;
	if (mythread_create(&t, test_routine, NULL))
	{
		fprintf(stderr, "Ошибка при создании потока\n");
		return 1;
	}

	sleep(1);
	return 0;
}
