#include "my_mutex.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

const int threads_num = 10;
const int iterations_num = 1e6;
volatile int shared_sum = 0;
my_mutex_t mutex;

void *thread_func(void *arg)
{
	int local_sum = 0;
	for (int i = 0; i < iterations_num; i++)
	{
		my_mutex_lock(&mutex);
		shared_sum += 1;
		my_mutex_unlock(&mutex);

		// if (!my_mutex_trylock(&mutex))
		// {
		// 	shared_sum += 1;
		// 	my_mutex_unlock(&mutex);
		// }
		// else
		// {
		// 	local_sum += 1;
		// }
	}

	my_mutex_lock(&mutex);
	shared_sum += local_sum;
	my_mutex_unlock(&mutex);

	printf("Thread %lld finished\n", (long long)arg);
	return NULL;
}

int main()
{
	pthread_t threads[threads_num];
	my_mutex_init(&mutex);

	for (int i = 0; i < threads_num; i++)
		pthread_create(&threads[i], NULL, thread_func, (void *)(long long)i);

	for (int i = 0; i < threads_num; i++)
		pthread_join(threads[i], NULL);

	printf("shared_sum = %d\n", shared_sum);

	my_mutex_destroy(&mutex);
	return 0;
}
