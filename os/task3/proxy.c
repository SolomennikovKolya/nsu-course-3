#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <curl/curl.h>
#include <netdb.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

const int PORT = 8080;			  // Порт прокси
const int BUFFER_SIZE = 8 * 1024; // По сколько байт читается из соединения
const int DEFAULT_DST_PORT = 80;  // Порт сервера назначения по умолчанию
const int LISTEN_BACKLOG = 10;	  // Максимальное количество подключений, которые могут ожидать принятия на сервере

#define MAX_HOST_LEN 256 // Максимальная длина хоста
#define MAX_PORT_LEN 6	 // Максимальная длина порта

const int SUCCESS = 0;	// Успех
const int FAILURE = -1; // Неудача

static int request_cnt = 0;

// Нужно чтобы передать аргументы в поточную функцию handle_client
typedef struct
{
	int client_socket;
} handle_client_args_t;

// Читает весь http GET запрос из сокета. При успехе возвращает строку запроса, иначе NULL
char *read_get_request(const int client_socket)
{
	char buffer[BUFFER_SIZE];
	char *request = NULL;
	ssize_t bytes_read;
	size_t total_read = 0;

	while ((bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0)) > 0)
	{
		// Выделяем нужное количество памяти под запрос
		char *new_request = realloc(request, total_read + bytes_read + 1);
		if (new_request == NULL)
		{
			fprintf(stderr, "read_get_request: failed to realloc\n");
			free(request);
			return NULL;
		}
		request = new_request;

		// Добавляем считанные данные в request
		buffer[bytes_read] = '\0';
		strncat(request, buffer, bytes_read);
		total_read += bytes_read;

		// Проверяем на завершение заголовков (два подряд идущих CRLF)
		if (strstr(request, "\r\n\r\n"))
			break;
	}

	if (bytes_read == -1)
	{
		fprintf(stderr, "handle_client: failed to read from client\n");
		free(request);
		return NULL;
	}

	return request;
}

// Проверяет, является ли запрос валидным
int is_valid_request(const char *request)
{
	if (request == NULL)
		return FAILURE;

	const int max_first_line_len = 1024 * 8 + 16 * 2;
	char first_line[max_first_line_len];

	char *first_line_end = strstr(request, "\r\n");
	if (first_line_end == NULL)
	{
		fprintf(stderr, "is_valid_request: failed to find the end of the first line\n");
		return FAILURE;
	}
	const int first_line_len = first_line_end - request;

	strncpy(first_line, request, first_line_len);
	first_line[first_line_len] = '\0';

	// Разделяем начальную строку на части
	char method[16], path[8192], version[16];
	if (sscanf(first_line, "%15s %8191s %15s", method, path, version) != 3)
	{
		fprintf(stderr, "is_valid_request: incorrect format of the initial line\n");
		return FAILURE;
	}

	// Проверяем метод
	if (strcmp(method, "GET") != 0)
	{
		fprintf(stderr, "is_valid_request: the %s method is not supported\n", method);
		return FAILURE;
	}

	// Проверяем версию HTTP
	if (strcmp(version, "HTTP/1.0") != 0 && strcmp(version, "HTTP/1.1") != 0)
	{
		fprintf(stderr, "is_valid_request: The %s protocol version is not supported\n", version);
		return FAILURE;
	}

	return SUCCESS;
}

// Извлекает хост и порт из URL
int addr_from_url(const char *buffer, char *host, char *port)
{
	// Парсим стартовую строку запроса (<Метод> <URL> <Версия HTTP>)
	char method[16], url[8192], protocol[16];
	sscanf(buffer, "%15s %2048s %16s", method, url, protocol);

	// Создаём объект URL
	CURLU *curl_url_obj = curl_url();
	if (!curl_url_obj)
	{
		fprintf(stderr, "addr_from_url: failed to create object CURLU\n");
		return FAILURE;
	}

	// Задаём URL для разбора
	CURLUcode res = curl_url_set(curl_url_obj, CURLUPART_URL, url, 0);
	if (res != CURLUE_OK)
	{
		fprintf(stderr, "addr_from_url: curl_url_set error: %s\n", curl_easy_strerror(res));
		curl_url_cleanup(curl_url_obj);
		return FAILURE;
	}

	// Получаем хост
	char *curl_host, *curl_port;
	if (curl_url_get(curl_url_obj, CURLUPART_HOST, &curl_host, 0) != CURLUE_OK)
	{
		fprintf(stderr, "addr_from_url: failed to get host from URL\n");
		return FAILURE;
	}
	strcpy(host, curl_host);
	curl_free(curl_host);

	// Получаем порт
	if (curl_url_get(curl_url_obj, CURLUPART_PORT, &curl_port, 0) != CURLUE_OK)
	{
		sprintf(port, "%d", DEFAULT_DST_PORT);
	}
	else
	{
		strcpy(port, curl_port);
		curl_free(curl_port);
	}

	curl_url_cleanup(curl_url_obj);
	return SUCCESS;
}

// Извлекает хост и порт из заголовков
int addr_from_headers(const char *buffer, char *host, char *port)
{
	// Ищем заголовок Host
	const char *host_header = "Host:";
	char *host_start = strstr(buffer, host_header);
	if (host_start == NULL)
	{
		fprintf(stderr, "addr_from_headers: failed to find Host header\n");
		return FAILURE;
	}

	// Пропускаем пробелы
	host_start += strlen(host_header);
	while (*host_start == ' ')
		host_start++;

	// Конец строки заголовка
	char *host_end = strstr(host_start, "\r\n");
	if (host_end == NULL)
	{
		fprintf(stderr, "addr_from_headers: failed to find end of Host header\n");
		return FAILURE;
	}

	// Парсим хост и порт
	char host_str[MAX_HOST_LEN] = {0};
	strncpy(host_str, host_start, host_end - host_start);
	char *port_start = strchr(host_str, ':');
	if (port_start)
	{
		strcpy(port, port_start + 1);
		*port_start = '\0';
	}
	else
	{
		sprintf(port, "%d", DEFAULT_DST_PORT);
	}
	strcpy(host, host_str);
	return SUCCESS;
}

// Извлекает адрес из URL или заголовков
int extract_addr(const char *buffer, char *host, char *port)
{
	if (addr_from_url(buffer, host, port) == SUCCESS)
		return SUCCESS;
	if (addr_from_headers(buffer, host, port) == SUCCESS)
		return SUCCESS;

	fprintf(stderr, "extract_addr: failed to extract address\n");
	return FAILURE;
}

// Подключение к серверу с адресом host:port
int connect_to_server(const char *host, const char *port)
{
	struct addrinfo hints; // Структура, которая содержит критерии для поиска адресов
	struct addrinfo *res;  // Указатель на результат, содержащий список подходящих адресов
	int server_socket;	   // Сокет целевого сервера

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;

	// Выполняет DNS-запрос или поиск в локальной конфигурации для разрешения имени хоста host и порта port
	if (getaddrinfo(host, port, &hints, &res) != 0)
	{
		fprintf(stderr, "connect_to_server: failed to resolve host\n");
		return FAILURE;
	}

	server_socket = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
	if (server_socket == -1)
	{
		fprintf(stderr, "connect_to_server: failed to create socket for server\n");
		freeaddrinfo(res);
		return FAILURE;
	}

	if (connect(server_socket, res->ai_addr, res->ai_addrlen) == -1)
	{
		fprintf(stderr, "connect_to_server: failed to connect to server\n");
		close(server_socket);
		freeaddrinfo(res);
		return FAILURE;
	}

	freeaddrinfo(res);
	return server_socket;
}

// Обработчик клиента
void handle_client(const int client_socket)
{
	// Читаем HTTP-запрос
	char *request = read_get_request(client_socket);
	if (request == NULL)
	{
		fprintf(stderr, "handle_client: failed to read request\n");
		return;
	}
	// printf("%s", request);

	if (is_valid_request(request) == FAILURE)
	{
		fprintf(stderr, "handle_client: invalid request\n");
		free(request);
		return;
	}

	// Извлекаем адрес назначения
	char dst_host[MAX_HOST_LEN] = "0";
	char dst_port[MAX_PORT_LEN] = "0";
	extract_addr(request, dst_host, dst_port);
	printf("dst_addr: %s:%s\n", dst_host, dst_port);

	// Подключаемся к целевому серверу
	int server_socket = connect_to_server(dst_host, dst_port);
	if (server_socket == -1)
	{
		fprintf(stderr, "handle_client: failed to connect to server\n");
		free(request);
		return;
	}

	// Пересылаем запрос клиента серверу
	if (send(server_socket, request, strlen(request), 0) == -1)
	{
		fprintf(stderr, "handle_client: failed to forward request to server\n");
		free(request);
		close(server_socket);
		return;
	}
	free(request);

	// Пересылаем ответ от сервера клиенту
	char buffer[BUFFER_SIZE];
	int bytes_read = 0;
	while (1)
	{
		bytes_read = recv(server_socket, buffer, sizeof(buffer), 0);
		if (bytes_read <= 0)
			break;

		if (send(client_socket, buffer, bytes_read, 0) == -1)
		{
			fprintf(stderr, "handle_client: failed to forward response to client\n");
			break;
		}

		if (strstr(buffer, "\r\n\r\n"))
			break;
	}
	if (bytes_read == -1)
		fprintf(stderr, "handle_client: error when reading data from the server\n");
	close(server_socket);
}

// Обертка для обработчика клиента
void *handle_client_wrapper(void *inp_args)
{
	printf("request %d: start\n", request_cnt);

	handle_client_args_t *args = (handle_client_args_t *)inp_args;
	int client_socket = args->client_socket;
	free(args);

	handle_client(client_socket);
	close(client_socket);

	printf("request %d: finish\n", request_cnt);
	request_cnt++;
	return NULL;
}

// Создание и настройка сокета для прокси сервера
int create_proxy_socket()
{
	int proxy_socket;
	struct sockaddr_in proxy_addr;

	// Создаем TCP сокет
	proxy_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (proxy_socket == -1)
	{
		fprintf(stderr, "create_proxy_socket: failed to create socket\n");
		return FAILURE;
	}

	// Заполняем структуру адреса прокси
	memset(&proxy_addr, 0, sizeof(proxy_addr));
	proxy_addr.sin_family = AF_INET;
	proxy_addr.sin_addr.s_addr = INADDR_ANY;
	proxy_addr.sin_port = htons(PORT);

	// Привязываем сокет к адресу
	if (bind(proxy_socket, (struct sockaddr *)&proxy_addr, sizeof(proxy_addr)) == -1)
	{
		close(proxy_socket);
		fprintf(stderr, "create_proxy_socket: bind failed\n");
		return FAILURE;
	}

	// Переводим сокет в режим прослушивания, чтобы он слушал входящие соединения
	if (listen(proxy_socket, LISTEN_BACKLOG) == -1)
	{
		close(proxy_socket);
		fprintf(stderr, "create_proxy_socket: listen failed\n");
		return FAILURE;
	}

	return proxy_socket;
}

int main()
{
	// const char *test_requests[] = {"GET http://api.ipify.org HTTP/1.0\r\nHost: api.ipify.org\r\n\r\n",			 // 1. api.ipify.org:80
	// 							   "GET http://api.ipify.org:8080 HTTP/1.0\r\nHost: api.ipify.org:8080\r\n\r\n", // 2. api.ipify.org:8080
	// 							   "GET /resource HTTP/1.1/1.1\r\nHost: example.com\r\n\r\n",					 // 3. example.com:80
	// 							   "GET http://192.168.1.1/resource HTTP/1.0\r\nHost: 192.168.1.1\r\n\r\n",		 // 4. 192.168.1.1:80
	// 							   "POST http://192.168.1.1:8080 HTTP/1.0\r\nHost: 192.168.1.1:8080\r\n\r\n",	 // 5. 192.168.1.1:8080
	// 							   "GET /resource HTTP/1.0\r\n\r\n",											 // 6. -
	// 							   "GET //example.com/resource HTTP/1.0\r\nHost: example.com"};					 // 7. -
	// for (int i = 0; i < 7; ++i)
	// {
	// 	// char dst_host[MAX_HOST_LEN] = "";
	// 	// char dst_port[MAX_PORT_LEN] = "";
	// 	// extract_addr(test_requests[i], dst_host, dst_port);
	// 	// printf("\ntest %d:\n", i + 1);
	// 	// printf("Host: %s\n", dst_host);
	// 	// printf("Port: %s\n", dst_port);

	// 	printf("\ntest %d:\n", i + 1);
	// 	if (is_valid_request(test_requests[i]) == SUCCESS)
	// 		printf("Valid request\n");
	// 	else
	// 		printf("Invalid request\n");
	// }
	// return 0;

	int proxy_socket = create_proxy_socket();
	if (proxy_socket == FAILURE)
		return FAILURE;
	printf("Proxy server is running on port %d\n", PORT);

	while (1)
	{
		// Принимаем соединение от клиента
		struct sockaddr_in client_addr;
		socklen_t client_addr_len = sizeof(client_addr);
		int client_socket = accept(proxy_socket, (struct sockaddr *)&client_addr, &client_addr_len);
		if (client_socket == -1)
		{
			fprintf(stderr, "main: accept failed\n");
			continue;
		}

		// Создаём структуру для аргументов обработчика клиента
		handle_client_args_t *args = malloc(sizeof(handle_client_args_t));
		if (!args)
		{
			fprintf(stderr, "main: failed to allocate memory for handle_client_args_t\n");
			close(client_socket);
			continue;
		}
		args->client_socket = client_socket;

		// Создаем поток для обработки клиента
		pthread_t thread;
		if (pthread_create(&thread, NULL, handle_client_wrapper, args) != 0)
		{
			fprintf(stderr, "main: failed to create thread\n");
			free(args);
			close(client_socket);
			continue;
		}
		pthread_detach(thread);
	}

	close(proxy_socket);
	return 0;
}
