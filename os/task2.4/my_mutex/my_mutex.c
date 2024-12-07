#include "my_mutex.h"

#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define SUCCESS 0			 // Успешное выполнение функции
#define ERR_INV_MUTEX -1	 // Неверный указатель на мьютекс
#define ERR_BUSY -2			 // Мьютекс занят
#define ERR_DOUBLE_UNLOCK -3 // Повторное освобождение
#define ERR_PERM -4			 // Неверный владелец мьютекса

static inline pid_t get_tid()
{
	return syscall(SYS_gettid);
}

// Переводит поток в состояние SLEEP, если *var_addr == expected_val
static inline int futex_wait(atomic_int *var_addr, const int expected_val)
{
	return syscall(SYS_futex, var_addr, FUTEX_WAIT, expected_val, NULL, NULL, 0);
}

// Пробуждает num_threads_to_wake потоков, ожидающих на данном футексе
static inline int futex_wake(atomic_int *var_addr, const int num_threads_to_wake)
{
	return syscall(SYS_futex, var_addr, FUTEX_WAKE, num_threads_to_wake, NULL, NULL, 0);
}

int my_mutex_init(my_mutex_t *mutex)
{
	if (!mutex)
		return ERR_INV_MUTEX;

	atomic_store(&mutex->state, 0);	 // Мьютекс свободен
	atomic_store(&mutex->owner, -1); // Нет владельца
	atomic_store(&mutex->futex, 0);

	return SUCCESS;
}

int my_mutex_destroy(my_mutex_t *mutex)
{
	if (!mutex)
		return ERR_INV_MUTEX;

	// Проверяем, что мьютекс свободен
	if (atomic_load(&mutex->state))
	{
		return ERR_BUSY;
	}

	return SUCCESS;
}

int my_mutex_lock(my_mutex_t *mutex)
{
	if (!mutex)
		return ERR_INV_MUTEX;

	const pid_t tid = get_tid();
	int expected = 0;

	// Пытаемся захватить мьютекс
	while (!atomic_compare_exchange_strong(&mutex->state, &expected, 1))
	{
		expected = 0; // Сброс ожидания для повторной проверки

		if (atomic_load(&mutex->owner) == tid)
		{
			return ERR_PERM;
		}

		// Если мьютекс заблокирован, ждём через futex
		if (futex_wait(&mutex->futex, 0) && errno != EAGAIN)
		{
			return errno;
		}
	}

	// Захватили мьютекс, устанавливаем владельца
	atomic_store(&mutex->owner, tid);

	return SUCCESS;
}

int my_mutex_trylock(my_mutex_t *mutex)
{
	if (!mutex)
		return ERR_INV_MUTEX;

	// Пытаемся захватить мьютекс
	int expected = 0;
	if (atomic_compare_exchange_strong(&mutex->state, &expected, 1))
	{
		atomic_store(&mutex->owner, get_tid());
		return SUCCESS;
	}

	return ERR_BUSY; // Мьютекс уже захвачен
}

int my_mutex_unlock(my_mutex_t *mutex)
{
	if (!mutex)
		return ERR_INV_MUTEX;

	// Проверяем, что мы не освобождаем мьютекс второй раз
	if (!atomic_load(&mutex->state))
	{
		atomic_store(&mutex->state, 0);
		return ERR_DOUBLE_UNLOCK;
	}

	// Проверяем, что текущий поток является владельцем
	if (atomic_load(&mutex->owner) != get_tid())
	{
		return ERR_PERM;
	}

	// Сбрасываем владельца и освобождаем мьютекс
	atomic_store(&mutex->owner, -1);
	atomic_store(&mutex->state, 0);

	// Пробуждаем один из ожидающих потоков
	if (!futex_wake(&mutex->futex, 1))
	{
		return errno;
	}

	return SUCCESS;
}
