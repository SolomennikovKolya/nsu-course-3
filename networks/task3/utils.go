package main

import (
	"fmt"
	"io"
	"net/http"
)

func getBytesFromURL(url string, params map[string]string) ([]byte, error) {
	// Составление запроса
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return make([]byte, 0), fmt.Errorf("ошибка при создании GET запроса: %w", err)
	}

	// Обновляем RawQuery с новыми параметрами из quary
	query := request.URL.Query()
	for key, value := range params {
		query.Add(key, value)
	}
	request.URL.RawQuery = query.Encode()

	// Отправка запроса + получение ответа
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return make([]byte, 0), fmt.Errorf("ошибка при отправки GET запроса: %w", err)
	}
	defer response.Body.Close()

	// Проверка статуса
	if response.StatusCode != http.StatusOK {
		return make([]byte, 0), fmt.Errorf("ошибка: %s", response.Status)
	}

	// Читаем тело ответа
	responseBytes, err := io.ReadAll(response.Body)
	if err != nil {
		return make([]byte, 0), fmt.Errorf("ошибка при чтении байтов из responseBytes: %w", err)
	}

	return responseBytes, nil
}
