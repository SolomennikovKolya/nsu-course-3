#include "uthread.h"
#include <stdio.h>
#include <unistd.h>

void *test_start_func(void *arg)
{
	printf("func: Hello from thread %d\n", uthread_self_tid());
	while (1)
	{
		// if (uthread_self_tid() == 2)
		// {
		// 	// uthread_yield();
		// 	uthread_exit();
		// }
		sleep(0.01);
	}
}

int main()
{
	printf("main: Hello\n");

	uthread_t uthread1, uthread2;
	uthread_create(&uthread1, test_start_func, NULL);
	uthread_create(&uthread2, test_start_func, NULL);

	while (1)
	{
		sleep(1);
	}

	return 0;
}
