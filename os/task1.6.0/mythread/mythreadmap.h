#ifndef MYTHREADMAP_H
#define MYTHREADMAP_H

typedef struct tid_and_thread
{
	int tid;
	void *thread;
} tid_and_thread;

/* Словарь хранящий пары {tid : mythread_t}
По факту представляет собой односвязный список */
typedef struct mythreadmap
{
	tid_and_thread data;
	struct mythreadmap *next;
} mythreadmap;

void mythreadmap_push(mythreadmap **head_ref, const int tid, const void *const thread); // Добавление узла в конец списка
void *mythreadmap_get(mythreadmap *head, const int tid);								// Получить mythread_t по tid
void mythreadmap_delete(mythreadmap **head_ref, const int tid);							// Удаление узла по значению tid
int mythreadmap_len(mythreadmap *head);													// Длина списка (количество потоков)
void mythreadmap_print(mythreadmap *mythreadmap);										// Печать всего списка

#endif
