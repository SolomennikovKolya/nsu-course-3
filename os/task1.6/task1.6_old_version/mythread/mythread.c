#include "mythread.h"
// #include "mythreadmap.h"

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

// static mythreadmap *all_threads = NULL;

/* Выделение памяти для потока по следующей схеме:
[mythread_struct_t][регион без прав][стек] */
static void *create_stack(int id)
{
	char stack_filename[30];
	snprintf(stack_filename, sizeof(stack_filename), "mythread/stack-%d", id);

	int stack_fd = open(stack_filename, O_RDWR | O_CREAT, 0660);
	if (stack_fd == -1)
	{
		perror("Ошибка открытия файла для ассоциации со стеком");
		return NULL;
	}
	if (ftruncate(stack_fd, STACK_SIZE) == -1)
	{
		perror("Ошибка изменения размера файла стека");
		close(stack_fd);
		return NULL;
	}

	void *stack = mmap(NULL, STACK_SIZE, PROT_NONE, MAP_SHARED, stack_fd, 0);
	close(stack_fd);
	if (stack == MAP_FAILED || stack == NULL)
	{
		perror("Ошибка при выделении нового региона для стека");
		return NULL;
	}

	int err = mprotect(stack + PAGE_SIZE, STACK_SIZE - PAGE_SIZE, PROT_READ | PROT_WRITE);
	if (err == -1)
	{
		fprintf(stderr, "Ошибка при изменении прав региона\n");
		return NULL;
	}
	err = mprotect(stack, sizeof(mythread_struct_t), PROT_READ | PROT_WRITE);
	if (err == -1)
	{
		fprintf(stderr, "Ошибка при изменении прав региона\n");
		return NULL;
	}
	memset(stack, 0, STACK_SIZE);
	// memset(stack + sizeof(mythread_struct_t), 64, PAGE_SIZE - sizeof(mythread_struct_t));
	/* Фан факт: в /proc/pid/maps будет виден только 1 регион памяти, а не 3,
	так как ядро видит весь регион как одну область, привязанную к файлу */

	return stack;
}

// Освобождение стека
static void free_stack(void *stack)
{
	const unsigned long struct_size = sizeof(mythread_struct_t);
	if (munmap(stack, struct_size) == -1)
	{
		perror("Ошибка при освобождении региона для данных структуры");
	}
}

// Обёртка над начальной функцией потока
static int start_routine_wrapper(void *thread_iter)
{
	mythread_t thread = (mythread_t)thread_iter;
	void *(*start_routine)(void *) = thread->start_routine;
	void *arg = thread->arg;
	// getcontext(&(thread->before_start_routine));

	start_routine(arg);

	// if (!thread->canceled)
	// {
	// 	start_routine(arg);
	// }
	// thread->finished = 1;
	// while (!thread->joined)
	// {
	// 	sleep(1);
	// }

	free_stack(thread->stack);
	return 0;
}

// Создание нового потока
int mythread_create(mythread_t *thread_res, void *(*start_routine)(void *), void *arg)
{
	void *stack = create_stack(0);
	if (stack == NULL)
	{
		perror("Ошибка при создании стека");
		return FAILURE;
	}

	mythread_t thread = (mythread_t)(stack);
	thread->start_routine = start_routine;
	thread->arg = arg;
	thread->stack = stack;

	int tid = clone(start_routine_wrapper, stack + STACK_SIZE,
					CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM, thread);
	if (tid == -1)
	{
		perror("Ошибка clone() при создании потока");
		free_stack(stack);
		return FAILURE;
	}

	thread->pid = getpid();
	thread->tid = tid;

	// mythreadmap_push(&all_threads, tid, thread);
	*thread_res = thread;

	return SUCCESS;
}

int mythread_self(void)
{
	// int tid = syscall(SYS_gettid);
	// return mythreadmap_get(all_threads, tid);
	return syscall(SYS_gettid);
}
