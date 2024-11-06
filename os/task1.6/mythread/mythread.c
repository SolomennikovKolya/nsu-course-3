#include "mythread.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/sched.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define PAGE_SIZE 4096 // 4 КБ (одна страница)
#define STACK_SIZE (PAGE_SIZE * 10)
#define SUCCESS 0
#define FAILURE -1

// Обёртка над начальной функцией потока
static int start_routine_wrapper(void *thread_iter)
{
	mythread_t thread = (mythread_t)thread_iter;
	void *(*start_routine)(void *) = thread->start_routine;
	void *arg = thread->arg;

	// getcontext(&(thread->before_start_routine));

	thread->retval = start_routine(arg);

	free(thread->stack);
	free(thread);
	return 0;
}

int mythread_create(mythread_t *thread_res, void *(*start_routine)(void *), void *arg)
{
	void *stack = malloc(PAGE_SIZE);
	if (stack == NULL)
	{
		fprintf(stderr, "Ошибка выделения памяти для скета потока\n");
		return FAILURE;
	}

	mythread_t thread = malloc(sizeof(mythread_struct_t));
	if (thread == NULL)
	{
		fprintf(stderr, "Ошибка выделения памяти для управляющей структуры потока\n");
		free(stack);
		return FAILURE;
	}
	thread->start_routine = start_routine;
	thread->arg = arg;
	thread->stack = stack;

	int tid = clone(start_routine_wrapper, stack + STACK_SIZE, CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM, thread);
	if (tid == -1)
	{
		perror("Ошибка clone при создании потока");
		free(stack);
		free(thread);
		return FAILURE;
	}

	thread->pid = getpid();
	thread->tid = tid;

	*thread_res = thread;
	return SUCCESS;
}

int pthread_join(mythread_t thread, void **ret)
{
	if (thread->joined)
	{
		fprintf(stderr, "Ошибка при присоединении потока. Поток уже был присоединён\n");
		return FAILURE;
	}
	return SUCCESS;
}