#include "uthread.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#define PAGE_SIZE sysconf(_SC_PAGESIZE) // Размер страницы системы (обычно 4Кб)
#define STACK_SIZE (PAGE_SIZE * 10)		// Размер стека для потока
#define MAX_UTHREADS_NUM 10				// Максимальное количество потоков
#define SWITCH_DT 1000000 - 1			// Время между вытеснениями потоков (в микросекундах)
#define SUCCESS 0
#define FAILURE -1

// Общие данные для работы потоков
static struct shared_uthread_data_t
{
	uthread_t uthreads[MAX_UTHREADS_NUM]; // Массив всех потоков (uthreads[0] - фиктивная запись под main поток)
	int cur_tid;						  // Идентификатор текущего потока (0 означает main поток)
	int uthreads_num;					  // Количество запущенных потоков (включая main поток)
	ucontext_t main_context;			  // Контекст main потока
	int need_to_clean;					  // Нужно ли освободить память замершённых потоков
} data;

static int data_initialized = 0;

void handle_sigalrm(int signo);

// Первичная настройка ресурсов библиотеки (выполняется только 1 раз)
static void launch_lib()
{
	if (data_initialized)
		return;

	// Заполнение общих данных
	memset(&data, 0, sizeof(data));
	data.cur_tid = 0; // по логике в эту функцию может зайти только main поток
	data.uthreads_num = 1;
	data.need_to_clean = 0;

	struct sigaction sa;
	struct itimerval timer;
	sigset_t mask, old_mask;

	// Настройка обработчика сигнала SIGALRM
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = handle_sigalrm;
	sa.sa_flags = 0; // Флаги (0 = обычное поведение)
	if (sigaction(SIGALRM, &sa, NULL) == -1)
	{
		perror("sigaction");
		exit(EXIT_FAILURE);
	}

	// Блокируем все сигналы, кроме SIGALRM и SIGINT
	sigfillset(&mask);
	sigdelset(&mask, SIGALRM);
	sigdelset(&mask, SIGINT);
	if (sigprocmask(SIG_SETMASK, &mask, &old_mask) == -1)
	{
		perror("sigprocmask");
		exit(EXIT_FAILURE);
	}

	// Настройка таймера
	timer.it_value.tv_sec = 0;
	timer.it_value.tv_usec = SWITCH_DT; // Первый запуск
	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = SWITCH_DT; // Периодичность

	if (setitimer(ITIMER_REAL, &timer, NULL) == -1)
	{
		perror("setitimer");
		exit(EXIT_FAILURE);
	}

	data_initialized = 1;
}

/* Округляет `size` вверх до размера, кратного размеру страницы, и добавляет одну дополнительную страницу.
Это нужно, чтобы была как мимнимум одна страница для самого стека и одна защищённая страница */
static size_t calculate_stack_size(const size_t size)
{
	if (size == 0)
		return PAGE_SIZE * 2;
	else
		return ((size + PAGE_SIZE - 1) / PAGE_SIZE + 1) * PAGE_SIZE;
}

// Cоздание стека
static void *create_stack(size_t size)
{
	size = calculate_stack_size(size);

	// Выделяем память под стек
	void *stack = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN, -1, 0);
	if (stack == MAP_FAILED)
	{
		perror("uthread: create_stack: mmap failed");
		return NULL;
	}

	// Создаём "охранную страницу" в нижней части стека
	if (mprotect(stack, PAGE_SIZE, PROT_NONE) != 0)
	{
		perror("uthread: create_stack: mprotect failed");
		munmap(stack, size);
		return NULL;
	}

	return stack;
}

// Освобождения памяти стека
static void free_stack(void *stack, size_t size)
{
	if (stack && size > 0)
	{
		size = calculate_stack_size(size);
		munmap(stack, size);
	}
}

/* Функция переключения потоков.
`next_tid` - поток, на который надо переключиться.
Если next_tid < 0 || next_tid > MAX_UTHREADS_NUM - 1, то планировщик сам выбирает следующий поток */
void schedule(int next_tid)
{
	if (data.uthreads_num < 2)
	{
		// Если потоков меньше двух, то переключаться не надо
		return;
	}

	const int cur_tid = data.cur_tid;

	// Выбор следующего потока
	if (next_tid < 0 || next_tid > MAX_UTHREADS_NUM - 1)
	{
		next_tid = (cur_tid + 1) % MAX_UTHREADS_NUM;
		for (int q = 0; q < MAX_UTHREADS_NUM; ++q)
		{
			if (next_tid == 0 || (data.uthreads[next_tid] && !data.uthreads[next_tid]->finished && next_tid != cur_tid))
				break;
			next_tid = (next_tid + 1) % MAX_UTHREADS_NUM;
		}
		if (next_tid == cur_tid)
		{
			return;
		}
	}

	// Переключение контекста
	ucontext_t *cur_ctx = (cur_tid == 0 ? &data.main_context : &data.uthreads[cur_tid]->context);
	ucontext_t *next_ctx = (next_tid == 0 ? &data.main_context : &data.uthreads[next_tid]->context);
	data.cur_tid = next_tid;
	printf("переключение: %d -> %d\n", cur_tid, data.cur_tid);
	swapcontext(cur_ctx, next_ctx);

	// Освобождение памяти для завершённых потоков
	if (data.need_to_clean)
	{
		for (int i = 1; i < MAX_UTHREADS_NUM; ++i)
		{
			if (data.uthreads[i] && data.uthreads[i]->finished)
			{
				free_stack(data.uthreads[i]->stack, data.uthreads[i]->stack_size);
				free(data.uthreads[i]);
				data.uthreads[i] = NULL;
				data.uthreads_num--;
				printf("Освобождён поток %d\n", i);
			}
		}
		data.need_to_clean = 0;
	}
}

// Обработчик сигнала SIGALRM, который вытесняет потоки
void handle_sigalrm(int signo)
{
	if (signo == SIGALRM)
	{
		// printf("Received SIGALRM\n");
		schedule(-1);
	}
}

// Обёртка над стартовой функцией потока
static void routine_wrapper(int cur_tid)
{
	data.uthreads[cur_tid]->routine(data.uthreads[cur_tid]->arg);
	data.uthreads[cur_tid]->finished = 1;
	data.need_to_clean = 1;
}

int uthread_create(uthread_t *uthread, void *(*routine)(void *), void *arg)
{
	launch_lib();

	if (data.uthreads_num == MAX_UTHREADS_NUM)
	{
		perror("uthread: uthread_create: the limit on the number of threads has been reached");
		return FAILURE;
	}

	// Создаём управляющую структуру для нового потока
	uthread_t new_uthread = malloc(sizeof(uthread_data_t));
	if (!new_uthread)
	{
		perror("uthread: uthread_create: malloc failed");
		return FAILURE;
	}
	new_uthread->stack_size = STACK_SIZE;
	new_uthread->stack = create_stack(new_uthread->stack_size);
	if (new_uthread->stack == NULL)
	{
		free(new_uthread);
		return FAILURE;
	}
	new_uthread->routine = routine;
	new_uthread->arg = arg;

	// Добавляем new_uthread в массив всех потоков
	int new_tid = -1;
	for (int i = 1; i < MAX_UTHREADS_NUM; ++i)
	{
		if (data.uthreads[i] == NULL)
		{
			new_tid = i;
			break;
		}
	}
	if (new_tid == -1)
	{
		// До сюда программа дойти не должна, но на всякий случай)
		perror("uthread: uthread_create: error when adding a new stream to the array");
		free_stack(new_uthread->stack, new_uthread->stack_size);
		free(new_uthread);
		return FAILURE;
	}
	data.uthreads[new_tid] = new_uthread;
	data.uthreads_num++;

	// Создаем контекст для нового потока
	getcontext(&new_uthread->context);
	new_uthread->context.uc_stack.ss_size = new_uthread->stack_size;
	new_uthread->context.uc_stack.ss_sp = new_uthread->stack;
	new_uthread->context.uc_link = &data.main_context;
	makecontext(&new_uthread->context, (void (*)(void))routine_wrapper, 1, new_tid);

	*uthread = new_uthread;

	// Запуск созданного потока
	schedule(new_tid);

	return SUCCESS;
}

uthread_t uthread_self(void)
{
	launch_lib();
	return data.uthreads[data.cur_tid];
}

int uthread_self_tid(void)
{
	launch_lib();
	return data.cur_tid;
}

void uthread_exit(void)
{
	launch_lib();

	if (data.cur_tid != 0)
	{
		data.uthreads[data.cur_tid]->finished = 1;
		data.need_to_clean = 1;
		schedule(-1);
	}
	else
	{
		exit(0);
	}
}

void uthread_yield(void)
{
	launch_lib();
	schedule(-1);
}
