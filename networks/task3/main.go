package main

import (
	"log"
	"os"
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
	openweatherURL               = "https://api.openweathermap.org/data/2.5/weather"
	openweatherKEY               = "688ffe0da9e16360228d1cb034013087"
	kudagoURL                    = "https://kudago.com/public-api/v1.2/places/"
)

type Data struct {
	locations        []Location // Список локаций
	selectedLocation int        // Номер локации в locations, уоторая сейчас выбрана
	weather          Weather    // Погода в выбранной локации
	places           []Place    // Интересные места
	mutex            sync.Mutex
}

func (data *Data) getSelectedLocationName() string {
	if data.selectedLocation != -1 {
		return data.locations[data.selectedLocation].Name
	} else {
		return ""
	}
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
	data := Data{locations: make([]Location, maxLocationsNum), selectedLocation: -1}

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
							go getLocations(e.Text, &data)
							locationNameInput.SetText("")
						}
						return material.Editor(theme, locationNameInput, defaultInputLocationNameText).Layout(gtx)
					}),

					// Список найденных локаций
					layout.Rigid(func(gtx layout.Context) layout.Dimensions {
						return material.List(theme, locationsList).Layout(gtx, maxLocationsNum, func(gtx layout.Context, i int) layout.Dimensions {
							data.mutex.Lock()
							locationName := data.locations[i].Name
							buttonText := getLocationText(data.locations[i])
							data.mutex.Unlock()

							if locationName != "" {
								btn := material.Button(theme, locationButtons[i], buttonText)
								if locationButtons[i].Clicked(gtx) {
									data.selectedLocation = i
									go getWeather(&data)
									go getPlaces(&data)
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
						data.mutex.Lock()
						fullText += "Локация: " + data.getSelectedLocationName() + "\n"
						fullText += "Погода: " + weatherToText(data.weather) + "\n"
						fullText += "Места: \n" + placesToText(data.places) + "\n"
						data.mutex.Unlock()

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

func run() {
	window := new(app.Window) // Создание нового окна:
	window.Option(app.Title("Places Finder"))
	window.Option(app.Size(unit.Dp(600), unit.Dp(800)))

	err := loop(window) // Эта функция должна выполняться бесконечно, пока программа работает
	if err != nil {
		log.Fatal(err) // Выводит ошибку в stderr (в формате время:ошибка) и делает os.Exit(1)
	} else {
		os.Exit(0) // Завершает всю программу (не только эту горутину)
	}
}

func main() {
	go run()   // Запуск приложения
	app.Main() // Запуск основного цикла событий
}
