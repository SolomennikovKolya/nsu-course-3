#ifndef UTHREAD_H
#define UTHREAD_H

#include <ucontext.h>

// Управляющая структура потока (данные для его работы)
typedef struct
{
	int tid;				  // Идентификатор потока
	void *(*routine)(void *); // Функция потока
	void *arg;				  // Аргументы для функции потока
	ucontext_t context;		  // Контекст выполнения потока
	long long stack_size;	  // Размер стека (в байтах)
	void *stack;			  // Указатель на начало стека
	int finished;			  // Завершена ли функция потока
} uthread_data_t;

// Указатель на управляющую структуру потока
typedef uthread_data_t *uthread_t;

/* Создание пользовательского потока
`uthread` - указатель на память, куда будет положен указатель на управляющую структуру потока
`routine` - начальная функция потока
`arg`     - аргументы для начальной функции
Возвращает: успех/неудача (0 либо -1) */
int uthread_create(uthread_t *uthread, void *(*routine)(void *), void *arg);

/* Возвращает идентификатор текущего потока */
uthread_t uthread_self(void);

/* Возвращает числовой идентификатор текущего потока */
int uthread_self_tid(void);

/* Завершает текущий поток */
void uthread_exit(void);

/* Уступает выполнение другому потоку */
void uthread_yield(void);

#endif
