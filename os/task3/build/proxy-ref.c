#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// http://lib.ru
// http://example.com
// http://httpstat.us/404

#define PORT 8080
#define BUFFER_SIZE 8292
#define CACHE_SIZE 8192

#define CACHE_LIFE_TIME 10
#define CACHE_CLEANUP_TIMEOUT 3
typedef struct CacheEntry
{
	char url[256];
	char *response;
	size_t size;
	size_t loaded_size;
	int is_loading;
	time_t last_access_time;
	pthread_cond_t cond;
	pthread_mutex_t mutex;
	int clients_waiting[10];
	int client_count;
} CacheEntry;

CacheEntry cache[CACHE_SIZE];
pthread_mutex_t cache_mutex = PTHREAD_MUTEX_INITIALIZER;

void *clean_cache_entry(CacheEntry *cache)
{
	printf("Cache entry cleaning...%s ", cache->url);
	strcpy(cache->url, "////");
	free(cache->response);
	cache->response = NULL;
	cache->size = 0;
	cache->loaded_size = 0;
	cache->is_loading = 0;
	cache->client_count = 0;
	printf("SUCCESS\n");
	return NULL;
}

void *cache_cleanup(void *arg)
{
	while (1)
	{
		sleep(CACHE_CLEANUP_TIMEOUT);
		printf("Scheduled cache checking...\n");
		time_t now = time(NULL);
		pthread_mutex_lock(&cache_mutex);
		for (int i = 0; i < CACHE_SIZE; i++)
		{
			if (cache[i].response != NULL && difftime(now, cache[i].last_access_time) > CACHE_LIFE_TIME)
			{
				clean_cache_entry(&cache[i]);
			}
		}
		pthread_mutex_unlock(&cache_mutex);
	}
	return NULL;
}

CacheEntry *find_in_cache(const char *url)
{
	pthread_mutex_lock(&cache_mutex);
	for (int i = 0; i < CACHE_SIZE; i++)
	{
		if (strcmp(cache[i].url, url) == 0)
		{
			cache[i].last_access_time = time(NULL);
			pthread_mutex_unlock(&cache_mutex);
			printf("Cache hit: %s\n", url);
			return &cache[i];
		}
	}
	pthread_mutex_unlock(&cache_mutex);
	printf("Cache miss: %s\n", url);
	return NULL;
}

CacheEntry *add_to_cache(const char *url)
{
	pthread_mutex_lock(&cache_mutex);
	for (int i = 0; i < CACHE_SIZE; i++)
	{
		if (cache[i].response == NULL)
		{
			strcpy(cache[i].url, url);
			cache[i].response = (char *)malloc(BUFFER_SIZE);
			cache[i].size = BUFFER_SIZE;
			cache[i].loaded_size = 0;
			cache[i].is_loading = 1;
			cache[i].last_access_time = time(NULL);
			cache[i].client_count = 0;
			pthread_mutex_init(&cache[i].mutex, NULL);
			pthread_cond_init(&cache[i].cond, NULL);
			printf("Caching started: %s\n", url);
			pthread_mutex_unlock(&cache_mutex);
			return &cache[i];
		}
	}
	pthread_mutex_unlock(&cache_mutex);
	return NULL;
}

void update_cache(CacheEntry *cache_entry, const char *data, size_t size)
{
	pthread_mutex_lock(&cache_entry->mutex);
	cache_entry->last_access_time = time(NULL);
	if (cache_entry->loaded_size + size > cache_entry->size)
	{
		cache_entry->response = (char *)realloc(cache_entry->response, cache_entry->loaded_size + size);
		cache_entry->size = cache_entry->loaded_size + size;
	}
	memcpy(cache_entry->response + cache_entry->loaded_size, data, size);
	cache_entry->loaded_size += size;
	pthread_cond_broadcast(&cache_entry->cond);
	pthread_mutex_unlock(&cache_entry->mutex);
}

void mark_cache_complete(CacheEntry *cache_entry)
{
	pthread_mutex_lock(&cache_entry->mutex);
	cache_entry->is_loading = 0;
	pthread_cond_broadcast(&cache_entry->cond);
	pthread_mutex_unlock(&cache_entry->mutex);
}

void *download_to_cache(void *cache_entry_ptr)
{
	CacheEntry *cache_entry = (CacheEntry *)cache_entry_ptr;

	char hostname[256];
	int port = 80;
	char path[256] = "/";

	if (sscanf(cache_entry->url, "http://%255[^:/]:%d/%255[^\n]", hostname, &port, path) < 3)
	{
		sscanf(cache_entry->url, "http://%255[^:/]/%255[^\n]", hostname, path);
	}

	printf("Host: %s, Path: %s, Port: %d\n", hostname, path, port);

	int server_socket;
	struct sockaddr_in server_addr;
	struct hostent *server;

	server = gethostbyname(hostname);
	if (server == NULL)
	{
		perror("Ошибка разрешения хоста");
		mark_cache_complete(cache_entry);
		clean_cache_entry(cache_entry);
		return NULL;
	}

	server_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (server_socket < 0)
	{
		perror("Ошибка создания сокета");
		mark_cache_complete(cache_entry);
		clean_cache_entry(cache_entry);
		return NULL;
	}

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
	server_addr.sin_port = htons(port);

	if (connect(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
	{
		perror("Ошибка подключения к серверу");
		close(server_socket);
		mark_cache_complete(cache_entry);
		clean_cache_entry(cache_entry);
		return NULL;
	}

	char request[BUFFER_SIZE];
	sprintf(request, "GET /%s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n", path, hostname);
	printf("REQUEST: %s", request);
	send(server_socket, request, strlen(request), 0);

	char response[BUFFER_SIZE];
	ssize_t bytes_read;
	int response_code;

	bytes_read = recv(server_socket, response, sizeof(response), 0);
	if (bytes_read <= 0)
	{
		perror("Ошибка чтения ответа от сервера");
		close(server_socket);
		mark_cache_complete(cache_entry);
		clean_cache_entry(cache_entry);
		return NULL;
	}

	sscanf(response, "HTTP/%*d.%*d %d", &response_code);
	if (response_code != 200)
	{
		printf("Ответ сервера с кодом %d, кэширование не будет выполнено\n", response_code);
		close(server_socket);
		mark_cache_complete(cache_entry);
		clean_cache_entry(cache_entry);
		return NULL;
	}

	update_cache(cache_entry, response, bytes_read);

	while ((bytes_read = read(server_socket, response, BUFFER_SIZE)) > 0)
	{
		update_cache(cache_entry, response, bytes_read);
	}

	close(server_socket);
	mark_cache_complete(cache_entry);

	return NULL;
}

int create_server_socket()
{
	int server_fd;
	struct sockaddr_in address;

	if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
	{
		perror("Socket creation failed");
		exit(EXIT_FAILURE);
	}

	int opt = 1;
	if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
	{
		perror("setsockopt(SO_REUSEADDR) failed");
		exit(EXIT_FAILURE);
	}

	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(PORT);

	if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
	{
		perror("Bind failed");
		exit(EXIT_FAILURE);
	}

	if (listen(server_fd, 10) < 0)
	{
		perror("Listen failed");
		exit(EXIT_FAILURE);
	}

	printf("Proxy server started on port %d\n", PORT);
	return server_fd;
}

void *handle_client(void *client_socket_ptr)
{
	int client_socket = *(int *)client_socket_ptr;
	free(client_socket_ptr);

	char buffer[BUFFER_SIZE];
	read(client_socket, buffer, BUFFER_SIZE);

	char method[16], url[256], protocol[16];
	sscanf(buffer, "%s %s %s", method, url, protocol);
	printf("Processing request: %s\n", url);

	CacheEntry *cache_entry = find_in_cache(url);
	if (cache_entry)
	{
		pthread_mutex_lock(&cache_entry->mutex);
		cache_entry->clients_waiting[cache_entry->client_count++] = client_socket;
		while (cache_entry->is_loading)
		{
			usleep(500000);
			for (int i = 0; i < cache_entry->client_count; i++)
			{
				send(cache_entry->clients_waiting[i], cache_entry->response, cache_entry->loaded_size, 0);
			}
			pthread_cond_wait(&cache_entry->cond, &cache_entry->mutex);
		}
		for (int i = 0; i < cache_entry->client_count; i++)
		{
			send(cache_entry->clients_waiting[i], cache_entry->response, cache_entry->loaded_size, 0);
		}
		pthread_mutex_unlock(&cache_entry->mutex);
		close(client_socket);
		return NULL;
	}

	cache_entry = add_to_cache(url);
	cache_entry->clients_waiting[cache_entry->client_count++] = client_socket;
	if (cache_entry == NULL)
	{
		perror("Не получилось добавить кэш entry");
		close(client_socket);
		return NULL;
	}

	pthread_t download_thread;
	pthread_create(&download_thread, NULL, download_to_cache, cache_entry);
	pthread_detach(download_thread);

	pthread_mutex_lock(&cache_entry->mutex);
	while (cache_entry->is_loading)
	{
		usleep(500000);
		for (int i = 0; i < cache_entry->client_count; i++)
		{
			send(cache_entry->clients_waiting[i], cache_entry->response, cache_entry->loaded_size, 0);
		}
		pthread_cond_wait(&cache_entry->cond, &cache_entry->mutex);
	}
	for (int i = 0; i < cache_entry->client_count; i++)
	{
		send(cache_entry->clients_waiting[i], cache_entry->response, cache_entry->loaded_size, 0);
	}

	pthread_mutex_unlock(&cache_entry->mutex);

	close(client_socket);
	return NULL;
}

int main()
{
	int server_fd = create_server_socket();

	pthread_t cleanup_thread;
	pthread_create(&cleanup_thread, NULL, cache_cleanup, NULL);
	pthread_detach(cleanup_thread);
	while (1)
	{
		struct sockaddr_in client_addr;
		socklen_t client_addr_len = sizeof(client_addr);
		int *client_socket = (int *)malloc(sizeof(int));

		*client_socket = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_len);
		if (*client_socket < 0)
		{
			perror("Accept failed");
			free(client_socket);
			continue;
		}

		printf("Client connected\n");

		pthread_t thread_id;
		pthread_create(&thread_id, NULL, handle_client, client_socket);
		pthread_detach(thread_id);
	}

	close(server_fd);
	return 0;
}
