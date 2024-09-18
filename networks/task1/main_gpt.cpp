#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>

using namespace std;

// Настройки времени для обновления и проверки "живых" узлов
const int heartbeatInterval = 2; // Интервал отправки сообщений в секундах
const int timeoutSeconds = 5;    // Таймаут для удаления неактивных копий

// Тип для хранения IP адресов активных копий
struct NodeInfo
{
    string ip;
    chrono::time_point<chrono::steady_clock> lastSeen;
};

void joinMulticastGroup(int sock, const string &multicastAddress, const sockaddr_in &groupAddr)
{
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = groupAddr.sin_addr.s_addr;
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);

    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&mreq, sizeof(mreq)) < 0)
    {
        perror("setsockopt");
        exit(1);
    }
}

void sendMessage(int sock, const sockaddr_in &groupAddr, const string &message)
{
    if (sendto(sock, message.c_str(), message.size(), 0, (struct sockaddr *)&groupAddr, sizeof(groupAddr)) < 0)
    {
        perror("sendto");
    }
}

string getIpAddress(const sockaddr_in &addr)
{
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(addr.sin_addr), buf, sizeof(buf));
    return string(buf);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <Multicast Address>\n";
        return 1;
    }

    string multicastAddress = argv[1];

    // Создание UDP сокета
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        perror("socket");
        return 1;
    }

    // Установка параметров адреса multicast-группы
    struct sockaddr_in groupAddr
    {
    };
    memset(&groupAddr, 0, sizeof(groupAddr));
    groupAddr.sin_family = AF_INET;
    groupAddr.sin_port = htons(12345); // Порт для обмена сообщениями

    // Преобразование multicast-адреса
    if (inet_pton(AF_INET, multicastAddress.c_str(), &groupAddr.sin_addr) <= 0)
    {
        cerr << "Invalid multicast address\n";
        return 1;
    }

    // Присоединение к multicast группе
    joinMulticastGroup(sock, multicastAddress, groupAddr);

    // Привязка сокета к порту
    struct sockaddr_in localAddr
    {
    };
    memset(&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_port = htons(12345); // Привязка к порту для приема сообщений
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0)
    {
        perror("bind");
        close(sock);
        return 1;
    }

    set<string> liveNodes;
    map<string, NodeInfo> nodeMap;

    auto lastSent = chrono::steady_clock::now();

    while (true)
    {
        // Проверка на отправку heartbeat сообщений
        auto now = chrono::steady_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - lastSent).count() >= heartbeatInterval)
        {
            sendMessage(sock, groupAddr, "ALIVE");
            lastSent = now;
        }

        // Проверка наличия сообщений
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(sock, &fds);

        struct timeval tv
        {
        };
        tv.tv_sec = 1; // Таймаут для select
        tv.tv_usec = 0;

        if (select(sock + 1, &fds, nullptr, nullptr, &tv) > 0)
        {
            // Принято сообщение
            struct sockaddr_in srcAddr
            {
            };
            socklen_t addrlen = sizeof(srcAddr);
            char buffer[256] = {0};

            int nbytes = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, (struct sockaddr *)&srcAddr, &addrlen);
            if (nbytes > 0)
            {
                string message(buffer, nbytes);
                string ip = getIpAddress(srcAddr);

                if (message == "ALIVE")
                {
                    nodeMap[ip] = {ip, chrono::steady_clock::now()};
                    liveNodes.insert(ip);
                }
            }
        }

        // Проверка таймаута для удаления "мертвых" узлов
        for (auto it = liveNodes.begin(); it != liveNodes.end();)
        {
            if (chrono::duration_cast<chrono::seconds>(now - nodeMap[*it].lastSeen).count() > timeoutSeconds)
            {
                cout << "Node " << *it << " is gone.\n";
                it = liveNodes.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // Вывод текущего списка живых узлов
        cout << "Live nodes: ";
        for (const auto &node : liveNodes)
        {
            cout << node << " ";
        }
        cout << endl;

        // Небольшая задержка перед следующим циклом
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    close(sock);
    return 0;
}

/*
ipconfig - посмотреть ip
*/