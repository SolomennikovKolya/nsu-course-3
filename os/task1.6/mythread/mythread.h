#ifndef MYTHREAD_H
#define MYTHREAD_H

#include <sys/types.h>
#include <ucontext.h>

// Управляющая структура потока
typedef struct
{
	void *(*start_routine)(void *arg_struct); // Указатель на стартовую функцию потока
	void *arg;								  // Аргументы для стартовой функции
	void *retval;							  // Результат стартовой функции
	void *stack;							  // Указатель на начало стека
	int mythread_id;						  // Идентификатор потока
	int pid;								  // Идентификатор процесса
	int finished;							  // Завершён поток или нет
	int joined;								  // Присоединён поток или нет
	int canceled;							  // Отменён поток или нет
	ucontext_t before_start_routine;		  // Контекст процессора перед запуском рутины
} mythread_struct_t;

// Указатель на управляющую структуру потока
typedef mythread_struct_t *mythread_t;

/* Создание своего ядерного потока
thread - указатель на память, куда будет положен идентификатор созданного потока
start_routine - начальная функция потока
arg - аргументы для начальной функции */
int mythread_create(mythread_t *thread, void *(*start_routine)(void *), void *arg);

#endif
