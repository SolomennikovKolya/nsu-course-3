#ifndef MY_MUTEX_H
#define MY_MUTEX_H

#include <stdatomic.h>

typedef struct __my_mutex_struct
{
	atomic_int state; // 0 - разблокирован, 1 - заблокирован
	atomic_int owner; // Владелец мьютекса: tid потока или -1, если нет владельца
	atomic_int futex; // futex для ожидания
} my_mutex_t;

int my_mutex_init(my_mutex_t *mutex);	 // Инициализация мьютекса
int my_mutex_destroy(my_mutex_t *mutex); // Уничтожение мьютекса
int my_mutex_lock(my_mutex_t *mutex);	 // Захват мьютекса
int my_mutex_trylock(my_mutex_t *mutex); // Попытка захвата мьютекса
int my_mutex_unlock(my_mutex_t *mutex);	 // Освобождение мьютекса

#endif
