package main

import (
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
)

// Парсит параметры командной строки
func parseFlags() (string, string, string, error) {
	var path, addr, port *string
	path = flag.String("path", "", "Path to the file to send")
	addr = flag.String("addr", "", "Server address")
	port = flag.String("port", "", "Server port")

	flag.Parse()

	var err error = nil
	if *path == "" || *addr == "" || *port == "" {
		err = errors.New("usage example: go run main.go -addr=\"127.0.0.1\" -port=\"8080\" -path=\"data/hello.txt\"")
		err = fmt.Errorf("ошибка при чтении параметров командной строки: %w", err)
	}

	return *path, *addr, *port, err
}

// Обмен данными с сервером
func dataExchange(conn net.Conn, filePath string) error {
	defer conn.Close()

	// Получение названия файла и размера названия
	buf := make([]byte, 2)
	filename := filepath.Base(filePath)
	filenameSize := uint16(len(filename))
	binary.BigEndian.PutUint16(buf, filenameSize)

	// Отправка размера названия файла
	_, err := conn.Write(buf)
	if err != nil {
		return fmt.Errorf("ошибка при отправке размера названия файла: %w", err)
	}

	// Отправка названия файла
	_, err = conn.Write([]byte(filename))
	if err != nil {
		return fmt.Errorf("ошибка при отправке названия файла: %w", err)
	}

	// Вычисление размера файла
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("ошибка при получении информации о файле: %w", err)
	}
	fileSize := uint64(fileInfo.Size())
	buf = make([]byte, 64)
	binary.BigEndian.PutUint64(buf, fileSize)

	// Отправка размера файла
	_, err = conn.Write(buf)
	if err != nil {
		return fmt.Errorf("ошибка при отправке размера файла: %w", err)
	}

	// Открытие файла
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("ошибка при открытии файла: %w", err)
	}
	defer file.Close()

	// Отправка файла (копирование данные из файла в соединение)
	_, err = io.Copy(conn, file)
	if err != nil {
		return fmt.Errorf("ошибка при отправке файла: %w", err)
	}

	// Приём результата обмена данными
	buf = make([]byte, 1)
	_, err = conn.Read(buf)
	if err != nil {
		return fmt.Errorf("ошибка при получении результата обмена данными: %w", err)
	}
	res := byte(buf[0])
	if res == 0 {
		fmt.Printf("файл %s успешно передан серверу\n", filename)
	} else {
		fmt.Printf("ошибка при отправке файла %s\n", filename)
	}

	return nil
}

// Логика работы клиента
func work() error {

	// Чтение параметров
	filePath, addr, port, err := parseFlags()
	if err != nil {
		return err
	}
	serverAddr := addr + ":" + port

	// Подключение к серверу
	conn, err := net.Dial("tcp", serverAddr)
	if err != nil {
		return fmt.Errorf("ошибка при подключении к серверу: %w", err)
	}
	fmt.Println("клиент подключился к серверу", serverAddr)

	// Обмен данными с сервером
	err = dataExchange(conn, filePath)
	if err != nil {
		return err
	}

	return nil
}

func main() {
	err := work()
	if err != nil {
		fmt.Println(err)
	}
}
