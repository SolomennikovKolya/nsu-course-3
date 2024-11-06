#include "mythreadmap.h"

#include <stdio.h>
#include <stdlib.h>

// Создание нового узла
static mythreadmap *create_mythreadmap(const int tid, const void *const thread)
{
	mythreadmap *new_mythreadmap = (mythreadmap *)malloc(sizeof(mythreadmap));
	if (!new_mythreadmap)
	{
		return NULL;
	}
	tid_and_thread data = {tid, thread};
	new_mythreadmap->data = data;
	new_mythreadmap->next = NULL;
	return new_mythreadmap;
}

void mythreadmap_push(mythreadmap **head_ref, const int tid, const void *const thread)
{
	mythreadmap *new_mythreadmap = create_mythreadmap(tid, thread);
	if (new_mythreadmap == NULL)
	{
		fprintf(stderr, "Ошибка при выделении памяти для новой ноды\n");
		return;
	}
	if (*head_ref == NULL)
	{
		*head_ref = new_mythreadmap;
	}
	else
	{
		mythreadmap *last = *head_ref;
		while (last->next != NULL)
		{
			last = last->next;
		}
		last->next = new_mythreadmap;
	}
}

void mythreadmap_delete(mythreadmap **head_ref, const int tid)
{
	mythreadmap *temp = *head_ref;
	mythreadmap *prev = NULL;

	if (temp != NULL && temp->data.tid == tid)
	{
		*head_ref = temp->next;
		free(temp);
		return;
	}

	while (temp != NULL && temp->data.tid != tid)
	{
		prev = temp;
		temp = temp->next;
	}

	if (temp == NULL)
		return;

	prev->next = temp->next;
	free(temp);
}

int mythreadmap_len(mythreadmap *head)
{
	int ans = 0;
	while (head != NULL)
	{
		head = head->next;
		ans++;
	}
	return ans;
}

// Возвращает остаток от деления a на base в пределах [0, base - 1]
static int mod(const int a, const int base)
{
	if (a < 0)
	{
		return a % base + base;
	}
	else
	{
		return a % base;
	}
}

void *mythreadmap_get(mythreadmap *head, const int tid)
{
	const int len = mythreadmap_len(head);
	if (len == 0)
	{
		fprintf(stderr, "Ошибка при получении элемента списка. Список пустой\n");
		return NULL;
	}

	for (int i = 0; i < mod(tid, len); ++i)
	{
		head = head->next;
	}
	return head->data.thread;
}

static void print_tid_and_thread(tid_and_thread d)
{
	printf("%d:%p", d.tid, d.thread);
}

void mythreadmap_print(mythreadmap *mythreadmap)
{
	while (mythreadmap != NULL)
	{
		print_tid_and_thread(mythreadmap->data);
		printf(" -> ");
		mythreadmap = mythreadmap->next;
	}
	printf("NULL\n");
}
