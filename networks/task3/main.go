package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"

	"image/color"

	"gioui.org/app"
	"gioui.org/layout"
	"gioui.org/op"
	"gioui.org/text"
	"gioui.org/unit"
	"gioui.org/widget"
	"gioui.org/widget/material"
)

const (
	maxLocationsNum              = 5
	maxPlacesNum                 = 5
	defaultInputLocationNameText = "Введите название локации..."
	graphhopperURL               = "https://graphhopper.com/api/1/geocode"
	graphhopperKEY               = "34e2aa3d-458f-4dec-978c-8f4045d876b6"
)

type Point struct {
	Lat float64 `json:"lat"`
	Lng float64 `json:"lng"`
}

type Location struct {
	Point       Point     `json:"point"`
	Extent      []float64 `json:"extent,omitempty"`
	Name        string    `json:"name"`
	Country     string    `json:"country"`
	City        string    `json:"city"`
	CountryCode string    `json:"countrycode"`
	State       string    `json:"state,omitempty"`
	Street      string    `json:"street"`
	Postcode    string    `json:"postcode,omitempty"`
	OsmID       int64     `json:"osm_id"`
	OsmType     string    `json:"osm_type"`
	OsmKey      string    `json:"osm_key"`
	OsmValue    string    `json:"osm_value"`
}

type Response struct {
	Hits   []Location `json:"hits"`
	Locale string     `json:"locale"`
}

// Получение вариантов локации по описанию
func getLocations(description string, locations []Location, mutex *sync.Mutex) {
	request, err := http.NewRequest("GET", graphhopperURL, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка при создании GET запроса:", err)
		return
	}
	query := request.URL.Query()
	query.Add("q", description)
	query.Add("limit", strconv.Itoa(maxLocationsNum))
	query.Add("key", graphhopperKEY)
	request.URL.RawQuery = query.Encode()

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка при отправки GET запроса:", err)
		return
	}
	defer response.Body.Close()

	responseBytes, err := io.ReadAll(response.Body)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка при чтении байтов из responseBytes:", err)
		return
	}

	var responseDecoded Response
	err = json.Unmarshal(responseBytes, &responseDecoded)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка при парсинге JSON:", err)
		return
	}

	if len(responseDecoded.Hits) == 0 {
		fmt.Fprintln(os.Stderr, "Нет результатов по данному описанию локации")
		return
	}

	mutex.Lock()
	copy(locations, responseDecoded.Hits)
	mutex.Unlock()
}

func locationToText(location *Location) string {
	ans := location.Name
	if location.Country != "" {
		ans += "\n" + location.Country
	}
	if location.City != "" {
		ans += "\n" + location.City
	}
	if location.State != "" {
		ans += "\n" + location.State
	}
	if location.Street != "" {
		ans += "\n" + location.Street
	}
	ans += "\n(" + strconv.FormatFloat(location.Point.Lat, 'f', -1, 64) + ", " + strconv.FormatFloat(location.Point.Lng, 'f', -1, 64) + ")"
	return ans
}

// Логика работы (обработка событий + отрисовка)
func loop(window *app.Window) error {
	// Тема (шрифт, цвета и т.д.)
	theme := material.NewTheme()
	theme.Palette.ContrastBg = color.NRGBA{R: 51, G: 173, B: 255, A: 255}

	// Контекст операций рисования ("журнал" команд, которые описывают, как элементы должны быть отрисованы в окне)
	var ops op.Ops

	// Виджет для ввода текста с клавиатуры:
	locationNameInput := new(widget.Editor)
	locationNameInput.Submit = true
	locationNameInput.SingleLine = true

	// Виджеты для выбора локации из списка
	locationsList := new(widget.List)
	locationsList.Axis = layout.Vertical
	var locationButtons [maxLocationsNum]*widget.Clickable
	for i := 0; i < maxLocationsNum; i++ {
		locationButtons[i] = new(widget.Clickable)
	}

	// Данные, которые могут изменяться из разных горутин
	locations := make([]Location, maxLocationsNum)
	selectedLocation := -1
	// var weather string                         // Погода в локации
	// var placeNames [maxPlacesNum]string        // Названия интересные места
	// var placeDescriptions [maxPlacesNum]string // Описания интересных мест
	var mutex sync.Mutex // Мьютекс для безопасного изменения и получения данных

	for {
		// .(type) - это type assertion (утверждение типа).
		// Позволяет проверить или получить конкретный тип значения, которое хранится в интерфейсе (интерфейс может быть реализован разными типами)
		switch event := window.Event().(type) {

		case app.DestroyEvent:
			return event.Err

		case app.FrameEvent:
			gtx := app.NewContext(&ops, event) // Обновление контекста

			inset := layout.UniformInset(unit.Dp(15))
			inset.Layout(gtx, func(gtx layout.Context) layout.Dimensions {
				return layout.Flex{Axis: layout.Vertical}.Layout(gtx,

					// Поле для ввода текста:
					layout.Rigid(func(gtx layout.Context) layout.Dimensions {
						inputEvent, _ := locationNameInput.Update(gtx)
						if e, ok := inputEvent.(widget.SubmitEvent); ok {
							go getLocations(e.Text, locations, &mutex)
							locationNameInput.SetText("")
							selectedLocation = -1
						}
						return material.Editor(theme, locationNameInput, defaultInputLocationNameText).Layout(gtx)
					}),

					// Список найденных локаций
					layout.Rigid(func(gtx layout.Context) layout.Dimensions {
						return material.List(theme, locationsList).Layout(gtx, maxLocationsNum, func(gtx layout.Context, i int) layout.Dimensions {
							mutex.Lock()
							locationName := locations[i].Name
							buttonText := locationToText(&locations[i])
							mutex.Unlock()

							if locationName != "" {
								btn := material.Button(theme, locationButtons[i], buttonText)
								if locationButtons[i].Clicked(gtx) {
									selectedLocation = i
									fmt.Printf("Выбрана локация: %s\n", buttonText)
								}
								return btn.Layout(gtx)
							} else {
								return material.Button(theme, locationButtons[i], "").Layout(gtx)
							}

						})
					}),

					// Разделитель между виджетами:
					layout.Rigid(
						layout.Spacer{Height: unit.Dp(10)}.Layout,
					),

					// Плашка с текстом
					layout.Rigid(func(gtx layout.Context) layout.Dimensions {
						fullText := ""
						if selectedLocation != -1 {
							mutex.Lock()
							fullText = "Локация: " + locations[selectedLocation].Name
							mutex.Unlock()
						}
						title := material.H6(theme, fullText)
						title.Alignment = text.Start
						return title.Layout(gtx)
					}),
				)
			})

			event.Frame(gtx.Ops) // Отрисовка кадра
		}
	}
}

// Запуск приложения
func run() {
	// Создание нового окна:
	window := new(app.Window)
	window.Option(app.Title("Places Finder"))
	window.Option(app.Size(unit.Dp(400), unit.Dp(600)))

	err := loop(window) // Эта функция должна выполняться бесконечно, пока программа работает
	if err != nil {
		log.Fatal(err) // Выводит ошибку в stderr (в формате время:ошибка) и делает os.Exit(1)
	} else {
		os.Exit(0) // Завершает всю программу (не только эту горутину)
	}
}

func main() {
	go run()
	app.Main() // Запуск основного цикла событий
}
