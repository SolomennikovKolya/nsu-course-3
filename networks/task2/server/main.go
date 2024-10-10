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
	"path/filepath"
	"sync"
	"time"
)

const (
	speedUpdateTime = 1
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

// CustomReader обертка вокруг io.Reader для подсчета байтов
type CustomReader struct {
	r         io.Reader
	bytesRead int64
	mutex     sync.Mutex
}

func (cr *CustomReader) Read(p []byte) (n int, err error) {
	n, err = cr.r.Read(p)
	cr.mutex.Lock()
	cr.bytesRead += int64(n)
	cr.mutex.Unlock()
	return
}

func (cr *CustomReader) BytesRead() int64 {
	cr.mutex.Lock()
	defer cr.mutex.Unlock()
	return cr.bytesRead
}

// Функция для форматирования байтов в человеко-читаемый вид
func formatBytes(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
		TB = GB * 1024
	)

	switch {
	case bytes >= TB:
		return fmt.Sprintf("%.2f TB", float64(bytes)/TB)
	case bytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/GB)
	case bytes >= MB:
		return fmt.Sprintf("%.2f MB", float64(bytes)/MB)
	case bytes >= KB:
		return fmt.Sprintf("%.2f KB", float64(bytes)/KB)
	default:
		return fmt.Sprintf("%d bytes", bytes)
	}
}

// Вывод скорости в консоль
func trackSpeed(cr *CustomReader, clientNum int, stopChan chan struct{}) {
	ticker := time.NewTicker(speedUpdateTime * time.Second)
	defer ticker.Stop()

	var prevBytesRead int64 = 0
	startTime := time.Now()

	for {
		select {
		case <-ticker.C:
			currentBytesRead := cr.BytesRead()
			bytesInLastInterval := currentBytesRead - prevBytesRead
			elapsedTime := time.Since(startTime).Seconds()

			fmt.Printf("Клиент: %d\n", clientNum)
			fmt.Printf("Мгновенная скорость: %s/сек\n", formatBytes(bytesInLastInterval/speedUpdateTime))
			fmt.Printf("Средняя скорость: %s/сек\n", formatBytes(int64(float64(currentBytesRead)/elapsedTime)))

			prevBytesRead = currentBytesRead

		case <-stopChan:
			elapsedTime := time.Since(startTime).Seconds()

			fmt.Printf("Клиент: %d\n", clientNum)
			if elapsedTime == 0 {
				fmt.Printf("Мгновенная скорость: бесконечность\n")
				fmt.Printf("Средняя скорость: бесконечность\n")
			} else if elapsedTime < speedUpdateTime {
				currentBytesRead := cr.BytesRead()
				fmt.Printf("Мгновенная скорость: %s/сек\n", formatBytes(int64(float64(currentBytesRead)/elapsedTime)))
				fmt.Printf("Средняя скорость: %s/сек\n", formatBytes(int64(float64(currentBytesRead)/elapsedTime)))
			}
			return
		}
	}
}

// Обработка клиента
func handleClient(conn net.Conn, clientNum int) {
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
	filenameToSave := "uploads/" + filepath.Base(filename)

	// Приём размера файла
	buf = make([]byte, 64)
	_, err = conn.Read(buf)
	if err != nil {
		fmt.Println("ошибка при чтении размера файла:", err)
		return
	}
	fileSize := int64(binary.BigEndian.Uint64(buf))

	// Открываем файл для записи полученных данных
	file, err := os.Create(filenameToSave)
	if err != nil {
		fmt.Println("ошибка при создании файла:", err)
		return
	}
	defer file.Close()

	// Приём файла (копирование данные из соединения в файл) + вывод скорости
	customReader := &CustomReader{r: conn, bytesRead: 0}
	limitedReader := io.LimitReader(customReader, fileSize)
	stopChan := make(chan struct{})
	go trackSpeed(customReader, clientNum, stopChan)

	_, err = io.Copy(file, limitedReader)
	if err != nil {
		fmt.Println("ошибка при приеме файла:", err)
		return
	}

	close(stopChan)

	// Вычисление размера полученного файла
	fileInfo, err := os.Stat(filenameToSave)
	if err != nil {
		fmt.Println("ошибка при получении информации о файле:", err)
		return
	}
	recievedFileSize := fileInfo.Size()

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

	for clientNum := 0; ; clientNum++ {
		// Ожидание подключения клиента
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("ошибка при принятии подключения:", err)
			continue
		}

		// Обработка клиента в отдельной горутине
		go handleClient(conn, clientNum)
	}
}

func main() {
	err := work()
	if err != nil {
		fmt.Println(err)
	}
}
