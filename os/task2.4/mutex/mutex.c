#include <linux/futex.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct
{
	_Atomic int lock;
} futex_mutex_t;

void futex_wait(int *futex, int val)
{
	syscall(SYS_futex, futex, FUTEX_WAIT, val, NULL, NULL, 0);
}

void futex_wake(int *futex)
{
	syscall(SYS_futex, futex, FUTEX_WAKE, 1, NULL, NULL, 0);
}

void futex_lock(futex_mutex_t *mutex)
{
	int expected = 0;
	while (!atomic_compare_exchange_weak(&mutex->lock, &expected, 1))
	{
		expected = 0;
		futex_wait(&mutex->lock, 1);
	}
}

void futex_unlock(futex_mutex_t *mutex)
{
	atomic_store(&mutex->lock, 0);
	futex_wake(&mutex->lock);
}
