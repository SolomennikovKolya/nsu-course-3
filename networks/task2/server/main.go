package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
)

// Парсит параметры командной строки
func parseFlags() (string, string, error) {
	var addr, port *string
	addr = flag.String("addr", "", "Server address")
	port = flag.String("port", "", "Server port")

	flag.Parse()

	var err error = nil
	if *addr == "" || *port == "" {
		err = errors.New("usage example: go run main.go -addr=\"127.0.0.1\" -port=\"8080\"")
		err = fmt.Errorf("ошибка при чтении параметров командной строки: %w", err)
	}

	return *addr, *port, err
}

// Обработка клиента
func handleClient(conn net.Conn) {
	defer conn.Close()

	// Приём размера названия файла
	var filenameSize int16
	buf := make([]byte, 2)
	_, err := conn.Read(buf)
	if err != nil {
		fmt.Println("ошибка при чтении размера названия файла:", err)
		return
	}
	filenameSize = int16(binary.BigEndian.Uint16(buf))

	// Приём названия файла
	reader := bufio.NewReader(conn)
	buf = make([]byte, filenameSize)
	_, err = reader.Read(buf)
	if err != nil {
		fmt.Println("ошибка при чтении названия файла:", err)
		return
	}
	filename := string(buf)
	filenameToSave := "uploads/" + filename

	// Приём размера файла
	buf = make([]byte, 64)
	_, err = conn.Read(buf)
	if err != nil {
		fmt.Println("ошибка при чтении размера файла:", err)
		return
	}
	fileSize := binary.BigEndian.Uint64(buf)

	// Открываем файл для записи полученных данных
	file, err := os.Create(filenameToSave)
	if err != nil {
		fmt.Println("ошибка при создании файла:", err)
		return
	}
	defer file.Close()

	// Приём файла (копирование данные из соединения в файл)
	limitedReader := io.LimitReader(conn, int64(fileSize))
	_, err = io.Copy(file, limitedReader)
	if err != nil {
		fmt.Println("ошибка при приеме файла:", err)
		return
	}

	// Получение размера файла
	fileInfo, err := os.Stat(filenameToSave)
	if err != nil {
		fmt.Println("ошибка при получении информации о файле:", err)
		return
	}
	recievedFileSize := uint64(fileInfo.Size())

	// Отправка результата обмена данными
	buf = make([]byte, 1)
	if recievedFileSize == fileSize {
		fmt.Printf("файл %s успешно получен\n", filename)
		buf[0] = 0
	} else {
		fmt.Printf("ошибка при получении файла %s\n", filename)
		buf[0] = 1
	}
	_, err = conn.Write(buf)
	if err != nil {
		fmt.Println("ошибка при отправке результата обмена данными:", err)
		return
	}
}

// Логика работы сервера
func work() error {

	// Чтение параметров
	addr, port, err := parseFlags()
	if err != nil {
		return err
	}
	var serverAddr string = addr + ":" + port

	// Запуск сервера на заданном адресе
	listener, err := net.Listen("tcp", serverAddr)
	if err != nil {
		return fmt.Errorf("ошибка при запуске сервера: %w", err)
	}
	defer listener.Close()
	fmt.Println("сервер запущен на", serverAddr)

	for {
		// Ожидание подключения клиента
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("ошибка при принятии подключения:", err)
			continue
		}

		// Обработка клиента в отдельной горутине
		go handleClient(conn)
	}
}

func main() {
	err := work()
	if err != nil {
		fmt.Println(err)
	}
}
