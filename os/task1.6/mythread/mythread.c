#include "mythread.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/sched.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define STACK_SIZE 1024 * 1024 // 64КБ
#define PAGE_SIZE 4096		   // 4Б (одна страница)
#define SUCCESS 0
#define FAILURE -1

// Создание стека для потока
void *create_stack(int id)
{
	char stack_filename[20];
	snprintf(stack_filename, sizeof(stack_filename), "stack-%d", id);

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

	// void *stack = mmap(NULL, STACK_SIZE, PROT_NONE, MAP_PRIVATE, stack_fd, 0);
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
		return NULL;
	}
	memset(stack + PAGE_SIZE, 0, STACK_SIZE - PAGE_SIZE);

	return stack;
}

// Освобождение стека
void free_stack(void *stack)
{
	// if (munmap(stack, STACK_SIZE) == -1)
	// {
	// 	perror("Ошибка при освобождении стека");
	// }
	printf("munmap success\n");
}

// Обёртка над начальной функцией потока
static int start_routine_wrapper(void *thread_iter)
{
	mythread_t thread = (mythread_t)thread_iter;
	void *(*start_routine)(void *) = thread->start_routine;
	void *arg = thread->arg;
	getcontext(&(thread->before_start_routine));

	// printf("star routine\n");
	start_routine(arg);
	// printf("finish routine\n");

	// if (!thread->canceled)
	// {
	// 	start_routine(arg);
	// }
	// thread->finished = 1;
	// while (!thread->joined)
	// {
	// 	sleep(1);
	// }

	free(thread);
	free_stack(thread->stack);

	return 0;
}

int mythread_create(mythread_t *thread_res, void *(*start_routine)(void *), void *arg)
{
	// Создаём стек для потока
	// thread->stack = malloc(STACK_SIZE);
	// if (thread->stack == NULL)
	// {
	// 	perror("Ошибка при создании стека");
	// 	free(thread);
	// 	return FAILURE;
	// }
	void *stack = create_stack(0);
	if (stack == NULL)
	{
		perror("Ошибка при создании стека");
		return FAILURE;
	}

	mythread_t thread = (mythread_t)stack;
	thread->start_routine = start_routine;
	thread->arg = arg;
	thread->stack = stack;

	/* Создаём новый поток
	CLONE_VM - общий адресное пространство памяти
	CLONE_FS - общий файловый системный контекст
	CLONE_FILES - общий набор открытых файлов
	CLONE_SIGHAND - общий обработчик сигналов
	CLONE_THREAD - определяет, что дочерний поток является частью того же процесса (потока)
	CLONE_SYSVSEM - разделение системных семафор SysV с родительским процессом */
	int pid = clone(start_routine_wrapper, (char *)stack + STACK_SIZE,
					CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM, thread);
	if (pid == -1)
	{
		perror("Ошибка clone() при создать поток");
		free_stack(stack);
		free(thread);
		return FAILURE;
	}

	thread->pid = pid;

	*thread_res = thread;

	return SUCCESS;
}
