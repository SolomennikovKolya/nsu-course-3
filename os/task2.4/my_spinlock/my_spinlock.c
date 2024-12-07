#include "my_spinlock.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#define SUCCESS 0  // Успешное выполнение функции
#define FAILURE -1 // Ошибка выполнения функции

int my_spin_init(my_spinlock_t *lock)
{
	if (!lock)
		return FAILURE;

	atomic_flag_clear(lock);
	return SUCCESS;
}

int my_spin_destroy(my_spinlock_t *lock)
{
	if (!lock)
		return FAILURE;

	return SUCCESS;
}

int my_spin_lock(my_spinlock_t *lock)
{
	if (!lock)
		return FAILURE;

	// atomic_flag_test_and_set - устанавливает атомарный флаг в true и возвращает его предыдущее значение
	while (atomic_flag_test_and_set(lock))
	{
	}
	return SUCCESS;
}

int my_spin_trylock(my_spinlock_t *lock)
{
	if (!lock)
		return FAILURE;

	if (!atomic_flag_test_and_set(lock))
		return SUCCESS;

	return FAILURE;
}

int my_spin_unlock(my_spinlock_t *lock)
{
	if (!lock)
		return FAILURE;

	atomic_flag_clear(lock);
	return SUCCESS;
}
