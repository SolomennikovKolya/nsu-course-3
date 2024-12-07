#ifndef MYSPINLOCK_H
#define MYSPINLOCK_H

#include <stdatomic.h>

typedef atomic_flag my_spinlock_t;

int my_spin_init(my_spinlock_t *lock);	  // Инициализация спинлока
int my_spin_destroy(my_spinlock_t *lock); // Уничтожение спинлока
int my_spin_lock(my_spinlock_t *lock);	  // Захват спинлока
int my_spin_trylock(my_spinlock_t *lock); // Попытка захвата спинлока
int my_spin_unlock(my_spinlock_t *lock);  // Освобождение спинлока

#endif
