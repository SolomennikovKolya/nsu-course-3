#include <arpa/inet.h> // Для sockaddr_in и inet_addr
#include <cstring>     // Для memset
#include <fcntl.h>     // Для функции fcntl
#include <iostream>
#include <map>
#include <string>
#include <sys/socket.h> // Для сокетов
#include <time.h>
#include <unistd.h> // Для close

const int PORT = 8080;                                 // Порт принимающего сокета (он же порт для мультикаст группы)
const double HEARTBEAT_DT = 1.0;                       // Время между отправками сообщений
const double TTL = 2.0;                                // Промежуток времени, пока найденные копии программы будут считаться живыми
const char *MESSAGE = "Hello, мир, manera крутит мир"; // Сообщение которое будет периодически отправляться
const int MESSAGE_LEN = strlen(MESSAGE);               // Длина сообщения

// Структурка для хранения мультикаст адреса (IPv4 или IPv6)
struct multicast_addr
{
    std::string str;
    int ip_version;
};

// Определяет, какой версии ip протокол. Возвращает AF_INET (если IPv4), AF_INET6 (если IPv6), либо -1 (если адрес не корректен)
int get_ip_version(const std::string &addr_str)
{
    struct in_addr addr;
    if (inet_pton(AF_INET, addr_str.c_str(), &addr) == 1)
        return AF_INET;

    struct in6_addr addr6;
    if (inet_pton(AF_INET6, addr_str.c_str(), &addr6) == 1)
        return AF_INET6;

    return -1;
}

// Проверяет, что мультикаст адрес действительно мультикаст
bool is_multicast(const multicast_addr &mult_addr)
{
    if (mult_addr.ip_version == AF_INET)
    {
        struct in_addr addr;
        inet_pton(AF_INET, mult_addr.str.c_str(), &addr);
        uint32_t ip = ntohl(addr.s_addr);
        return (ip >= 0xE0000000 && ip <= 0xEFFFFFFF);
    }
    else if (mult_addr.ip_version == AF_INET6)
    {
        struct in6_addr addr6;
        inet_pton(AF_INET6, mult_addr.str.c_str(), &addr6);
        return (addr6.s6_addr[0] == 0xFF);
    }
    else
        throw new std::runtime_error("mult_addr.ip_version должен быть либо AF_INET, либо AF_INET6");
}

// Считывание мультикаст адреса (IPv4 или IPv6) с проверкой на корректность
multicast_addr get_multicast_addr(int argc, char *argv[])
{
    if (argc != 2)
        throw new std::runtime_error("Предоставьте мультикаст адрес в аргументах программы");

    multicast_addr mult_addr;

    mult_addr.str = argv[1];

    mult_addr.ip_version = get_ip_version(mult_addr.str);
    if (mult_addr.ip_version == -1)
        throw new std::runtime_error("Неверный адрес");

    if (!is_multicast(mult_addr))
        throw new std::runtime_error("Адрес должен принадлежать мультикаст диапозону");

    return mult_addr;
}

// Создаёт и настраивает принимающий сокет
int setup_sock_in(const multicast_addr &mult_addr)
{
    // Создание сокета
    int sock_in = socket(AF_INET6, SOCK_DGRAM, 0);
    if (sock_in < 0)
        throw new std::runtime_error("Ошибка создания сокета");

    // Отключение флага IPV6_V6ONLY, чтобы сокет принимал и IPv4 и IPv6 (получится dual-stack сокет)
    int off = 0;
    if (setsockopt(sock_in, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off)) < 0)
        throw new std::runtime_error("Ошибка установки параметра IPV6_V6ONLY");

    // Настроука сокета на переиспользование адреса
    int reuse = 1;
    if (setsockopt(sock_in, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
        throw new std::runtime_error("Ошибка при установке параметров сокета");

    // Настройка сокета на неблокирующий режим
    int flags = fcntl(sock_in, F_GETFL, 0);
    if (flags == -1)
        throw new std::runtime_error("Ошибка при считывании флагов сокета");
    flags |= O_NONBLOCK;
    if (fcntl(sock_in, F_SETFL, flags) == -1)
        throw new std::runtime_error("Ошибка при попытке поставить неблокирующий режим для сокета");

    // Привязка сокета ко всем интерфейсам IPv6 и IPv4
    sockaddr_in6 sock_in_addr;
    memset(&sock_in_addr, 0, sizeof(sock_in_addr));
    sock_in_addr.sin6_family = AF_INET6;
    sock_in_addr.sin6_port = htons(PORT);
    sock_in_addr.sin6_addr = in6addr_any;

    if (bind(sock_in, (struct sockaddr *)&sock_in_addr, sizeof(sock_in_addr)) < 0)
        throw new std::runtime_error("Ошибка привязки сокета");

    // Присоединиться к multicast группе
    if (mult_addr.ip_version == AF_INET)
    {
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(mult_addr.str.c_str());
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
        if (setsockopt(sock_in, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0)
            throw new std::runtime_error("Ошибка при присоединении к multicast группе");
    }
    else if (mult_addr.ip_version == AF_INET6)
    {
        struct ipv6_mreq mreq6;
        memset(&mreq6, 0, sizeof(mreq6));
        inet_pton(AF_INET6, mult_addr.str.c_str(), &mreq6.ipv6mr_multiaddr);
        mreq6.ipv6mr_interface = 0; // Интерфейс по умолчанию (можно указать номер интерфейса)
        if (setsockopt(sock_in, IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &mreq6, sizeof(mreq6)) < 0)
            throw std::runtime_error("Ошибка при присоединении к multicast группе");
    }

    return sock_in;
}

// Создаёт и настраивает сокет для отправки
int setup_sock_out()
{
    // Создание сокета
    int sock_out = socket(AF_INET6, SOCK_DGRAM, 0);
    if (sock_out < 0)
        throw new std::runtime_error("Ошибка создания сокета");

    // Отключаем флаг IPV6_V6ONLY, чтобы сокет поддерживал как IPv4, так и IPv6
    int off = 0;
    if (setsockopt(sock_out, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off)) < 0)
        throw new std::runtime_error("Ошибка установки параметра IPV6_V6ONLY");

    return sock_out;
}

// Задаёт send_addr и send_addr_size для отправки сообщения
void get_send_addr(const multicast_addr &mult_addr, sockaddr_storage &send_addr, socklen_t &send_addr_size)
{
    if (mult_addr.ip_version == AF_INET)
    {
        struct sockaddr_in send_addr4;
        memset(&send_addr4, 0, sizeof(send_addr4));
        send_addr4.sin_family = AF_INET;
        send_addr4.sin_port = htons(PORT);
        send_addr4.sin_addr.s_addr = inet_addr(mult_addr.str.c_str());

        send_addr = *((struct sockaddr_storage *)&send_addr4);
        send_addr_size = sizeof(send_addr4);
    }
    else if (mult_addr.ip_version == AF_INET6)
    {
        struct sockaddr_in6 send_addr6;
        memset(&send_addr6, 0, sizeof(send_addr6));
        send_addr6.sin6_family = AF_INET6;
        send_addr6.sin6_port = htons(PORT);
        inet_pton(AF_INET6, mult_addr.str.c_str(), &send_addr6.sin6_addr);

        send_addr = *((struct sockaddr_storage *)&send_addr6);
        send_addr_size = sizeof(send_addr6);
    }
}

// Возвращает адреса отправителя в формате "ip : port"
std::string get_sender_addr_str(sockaddr_storage &sender_addr)
{
    if (sender_addr.ss_family == AF_INET)
    {
        char sender_ip[INET_ADDRSTRLEN];
        std::string sender_addr_str = "";
        sockaddr_in sender_addr4 = *((struct sockaddr_in *)&sender_addr);

        inet_ntop(AF_INET, &(sender_addr4.sin_addr), sender_ip, INET_ADDRSTRLEN);
        sender_addr_str += sender_ip;

        unsigned short sender_port = ntohs(sender_addr4.sin_port);
        sender_addr_str += " : " + std::to_string(sender_port);

        return sender_addr_str;
    }
    else if (sender_addr.ss_family == AF_INET6)
    {
        char sender_ip[INET6_ADDRSTRLEN];
        std::string sender_addr_str = "";
        sockaddr_in6 sender_addr6 = *((struct sockaddr_in6 *)&sender_addr);

        inet_ntop(AF_INET6, &(sender_addr6.sin6_addr), sender_ip, INET6_ADDRSTRLEN);
        sender_addr_str += sender_ip;

        unsigned short sender_port = ntohs(sender_addr6.sin6_port);
        sender_addr_str += " : " + std::to_string(sender_port);

        return sender_addr_str;
    }
    else
        throw new std::runtime_error("sender_addr.ss_family должен быть либо AF_INET, либо AF_INET6");
}

int main(int argc, char *argv[])
{
    int sock_in;                                     // Принимающий сокет
    int sock_out;                                    // Отправляющий сокет
    std::map<std::string, clock_t> live_copies = {}; // Живые копии программы. Хранит (адрес : время добавления в словарь)
    char buffer[MESSAGE_LEN + 1];                    // Буфер для получения сообщений

    try
    {
        // Адрес мультикаст группы
        multicast_addr mult_addr = get_multicast_addr(argc, argv);

        sock_in = setup_sock_in(mult_addr);
        sock_out = setup_sock_out();

        // Адрес сокета на который будет отправляться сообщение
        sockaddr_storage send_addr;
        socklen_t send_addr_size;
        get_send_addr(mult_addr, send_addr, send_addr_size);

        while (true)
        {
            if (sendto(sock_out, MESSAGE, MESSAGE_LEN, 0, (struct sockaddr *)&send_addr, send_addr_size) < 0)
                throw new std::runtime_error("Ошибка отправки сообщения");

            bool changed = false; // Чтобы отслеживать, были ли изменения за текущий цикл
            time_t heartbeat_start_time = clock();

            while (double(clock() - heartbeat_start_time) / CLOCKS_PER_SEC < HEARTBEAT_DT)
            {
                sockaddr_storage sender_addr;
                socklen_t sender_addr_len = sizeof(sender_addr);
                int recv_len = recvfrom(sock_in, buffer, sizeof(buffer), 0, (struct sockaddr *)&sender_addr, &sender_addr_len);

                if (recv_len < 0)
                {
                    if (errno == EWOULDBLOCK || errno == EAGAIN)
                        continue; // Данных пока нет
                    else
                        throw new std::runtime_error("Ошибка получения данных");
                }

                std::string sender_addr_str = get_sender_addr_str(sender_addr);

                if (live_copies.find(sender_addr_str) == live_copies.end())
                    changed = true;
                live_copies[sender_addr_str] = clock();
            }

            // Удаляем записи о живых копиях, у которых истекло время
            for (auto it = live_copies.begin(); it != live_copies.end();)
            {
                if (double(clock() - it->second) / CLOCKS_PER_SEC > TTL)
                {
                    it = live_copies.erase(it);
                    changed = true;
                }
                else
                    ++it;
            }

            if (changed)
            {
                std::cout << "Живых копий: " << live_copies.size() << "\n";
                for (auto it : live_copies)
                    std::cout << it.first << "\n";
            }
        }
    }
    catch (const std::runtime_error *e)
    {
        std::cerr << e->what() << std::endl;
        return 1;
    }

    close(sock_in);
    close(sock_out);
    return 0;
}
