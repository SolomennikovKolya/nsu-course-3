
### Полезности
- [Гайд на Go](https://metanit.com/go/tutorial/1.1.php)
- [Go cheatsheet](https://devhints.io/go)
- [Go dev tour](https://go.dev/tour/welcome/1)
- [Go с примерами](https://gobyexample.com)

### Команды
1. `go mod init <myproject>` - инициализирует модуть в текущей директории (создаёт файл go.mod)
2. `go get` - скачивает модуль, добавляет зависимости в go.mod, создаёт или обновляет файл go.sum
3. `go install` — скачивает, компилирует и установливает исполняемые файлов (если пакет с исходным кодом содкржит main) в директорию $GOPATH/bin (нужно для установки утилит и инструментов). Если в пакете нет функции main, команда go install не создаёт исполняемые файлы, но загружает их в ваш GOPATH или в кэш модулей.
4. `go mod tidy` - удаляет неиспользуемые зависимости в go.mod и добавляет недостающие (при необходимости скачивает недостающие пакеты).
5. `go run main.go` - сборка + запуск
6. `go run .` - запускает программу из точки входа (функция `main` в пакете `main`)

### Основные характеристики

**Go** (или Golang) — это компилируемый язык программирования, разработанный в Google в 2007 году и выпущенный в 2009 году. Основные аспекты языка:

- **Статическая типизация**: Типы данных в Go указываются явно, что позволяет избегать ошибок во время исполнения программы и делает код более предсказуемым.

- **Параллелизм**: Go поддерживает многозадачность с помощью лёгковесных потоков (goroutines), что делает его особенно подходящим для написания многопоточных приложений, таких как серверы и сетевые программы. Используются примитивы параллелизма Go, горутины и каналы. В основе модели параллелизма Go лежит планировщик, часть системы среды выполнения, которая управляет работой горутин. Планировщик Go является планировщиком M:N, потому что он распределяет M горутин на N потоков ОС, где M может быть намного больше N. 
	- **Горутины** - основная единицей процесса планирования в Go. Горутины дёшевы в создании и уничтожении, поскольку им требуется лишь небольшой объём памяти для их стека (по умолчанию стек горутины начинается всего с 2 КБ).
	- **Work stealing**: чтобы равномерно распределить рабочую нагрузку между несколькими потоками, Go использует жадную стратегию Work stealing. Каждый поток ОС поддерживает локальную очередь выполняемых горутин. Когда поток завершает выполнение своих локальных горутин, он пытается забрать простаивающие горутины других потоков. Это помогает держать все потоки занятыми и использовать все доступные ядра процессора.

- **Сборка мусора**: В Go встроен механизм автоматической сборки мусора (Garbage Collection), который работает конкурентно с выполнением программы на Go, что позволяет не останавливать выполнение программы для отчистки  мусора.

- **Встроенная поддержка работы с сетью**: Go изначально разрабатывался для высоконагруженных серверных приложений, поэтому в языке есть мощные средства для работы с сетевыми протоколами и сетевыми программами.

- **Быстрая компиляция**: Компилятор Go отличается высокой скоростью, что позволяет разработчикам быстро получать результаты своих изменений в коде. Проект можно статически скомпилировать в один исполняемый файл вместе со всеми используемыми библиотеками, что позволяет решить проблему с зависимостями. При сборке проекта Go все зависимости (модули) компилируются вместе с вашим кодом и встраиваются в один исполняемый файл. Это означает, что исполняемый файл будет содержать все необходимые библиотеки и зависимости, что делает его самодостаточным.

- **Стандартная библиотека**: Go имеет большую стандартную библиотеку, включающую пакеты для работы с файлами, сетями, криптографией, HTTP, JSON и многим другим.

- **Кросс-компиляция** — процесс компиляции кода на одном типе машины или операционной системы («хост») для запуска на другом типе машины или операционной системы («цель»). Она возможна потому, что Go имеет автономную, платформонезависимую стандартную библиотеку и среду выполнения, а значит, ему не нужно связываться с системными библиотеками, как это делают некоторые другие языки. Примеры компиляции на Linux, Windows и macOS с архитектурой amd64:
	- `GOOS=linux GOARCH=amd64 go build -o myprogram_linux myprogram.go`
	- `GOOS=windows GOARCH=amd64 go build -o myprogram_windows.exe myprogram.go`
	- `GOOS=darwin GOARCH=amd64 go build -o myprogram_mac myprogram.go`

- **Escape Analysis** — это техника оптимизации компилятора, которая определяет, можно ли безопасно выделить переменную в стеке, а не в куче, что может значительно повысить производительность.

- **Каналы** используются для безопасного обмена данными между горутинами. Это позволяет легко синхронизировать задачи и обмениваться данными без риска возникновения условий гонки. Вместо того, чтобы потоки обменивались данными через общую память, Go предлагает горутинам обмениваться данными через явные примитивы синхронизации — каналы.

- **Встроенные интсрументы**:
	- `go run` - компиляция + запуск (например `go run main.go`)
	- `go test` - запуск тестов
	- `go build` - компилирует код в .exe файл

- **Встроенный профилировщик** - для теста скорости

- **Система модулей Go** - модульность

### Типы данных

Типы данных можно разделить на ссылочные и не ссылочные. Не ссылочные типы представляют собой значения, которые копируются при передаче или присваивании.

#### 1. Не ссылочные (Value Types)
- *Примитивные типы*:
	- `int` - имеет размер 32 или 64 в зависимости от разрядности системы
	- `int8`, `int16`, `int32`, `int64` - фиксированный размер
	- `uint`, `uint8`, `uint16`, `uint32`, `uint64` - беззнаковые типы
    - `float32`, `float64` - как float и double в других языках
    - `complex64`, `complex128` - комплексные числа
    - `bool` - логический тип
    - `string` - строка
    - `byte` - байт (синоним типа uint8)
    - `rune` - unicode символ (синоним типа int32)
- *Типы структур*:
    - `struct`
- *Массивы и срезы*:
    - `array`
    - `slice`
- *Каналы*:
    - `chan`

#### 2. Ссылочные (Reference Types)
- `*T` - указатель на тип `T`
- `map[K]V` - карта (словарь)
- `chan T` канал
- `func` - функция

#### Особенности
- Переменные по умолчанию инициализируются "нулевым значением":
	- `int` → `0`
	- `string` → `""`
	- `bool` → `false`
	- Указатели → `nil`

### Факты
1. Программа начинает выполняться с пакета `main` (package main).
2. Магическая функция init() выполняется до main.
3. Управляющие структуры: if, for, switch, select, labels, range (slice, array, map)
4. Поддержка функций как объектов первого класса, то есть функции могут передаваться как аргументы, возвращаться из других функций и сохраняться в переменные. 
5. Анонимные функции и замыкания. Можно объявить функцию без имени и использовать переменные из внешнего контекста.
6. Операторы присвоения
	- "=" - присвоение значения уже объявленной переменной (например var x int = 10)
	- ":=" - объявление + присвоение (например x := 10)
7. `make` - команда для создания и инициализации срезов (slices), карт (maps) и каналов (channels). Это встроенная функция языка Go, и она отличается от команды `new`, которая просто выделяет память, но не инициализирует структуру данных.
8. `defer` откладывает выполнение функции до тех пор, пока текущая функция не завершится.

### Пакеты

**Пакет** — это единица кода в Go, которая может содержать функции, структуры и другие определения. Каждый пакет должен находиться в отдельной директории.
- Если имя переменной, функции или типа начинается с заглавной буквы, оно экспортируется и доступно для использования в других пакетах.
- Если имя начинается со строчной буквы, оно не экспортируется и доступно только в пределах текущего пакета.
- Имя пакета обычно соответствует имени папки, в которой он находится
- Все файлы одного пакета компилируются как единая сущность, поэтому переменные и функции в разных файлах одного пакета автоматически доступны без импорта.

### Модули
- **Модуль** — это группа связанных пакетов, которая может содержать один или несколько пакетов. Модуль имеет файл `go.mod`, который содержит метаданные о модуле, его зависимостях и версиях.
- Система модулей автоматически управляет зависимостями проекта, благодаря файлам `go.mod` (информация о модуле и его зависимостях и их версии) и `go.sum` (контрольные суммы всех зависимостей модуля для проверки целостности). Проект может располагаться где угодно на компьютере.
- Go-модули хранятся в локальном кэше в директории GOPATH/go/pkg/mod. Это позволяет не скачивать модули заного из интернета для каждого проекта.
- `GOPATH` — это путь, в котором раньше хранился исходный код Go-проектов, пакеты и собранные бинарные файлы (у меня это C:/Users/Пользователь).
- `GOROOT` — это путь к месту установки Go, где находятся стандартные библиотеки и сам компилятор (явно не указано, но у меня это C:\Program Files\Go).
- Про контрольные суммы в go.sum: Когда Go загружает модуль, он вычисляет контрольную сумму для содержимого этого модуля (включая код и метаданные). Эта контрольная сумма записывается в файл `go.sum`. Если модуль будет изменен или поврежден при последующей загрузке, контрольная сумма не совпадет с уже сохраненной, и Go выдаст ошибку.

### Где Go ищет зависимости при import?
1.  Стандартная библиотека Go
	- Пакеты, такие как `fmt`, `os`, `net/http`, и другие пакеты стандартной библиотеки ищутся в месте установки go
2. Локальные пакеты
	- Пакеты, которые находятся в том же модуле или проекте, ищутся в локальных директориях относительно вашей рабочей директории
3. Модули и сторонние (например `import "github.com/sirupsen/logrus"`) ищутся в:
	- Локальном кэше модулей (`$GOPATH/pkg/mod/`)
	- Репозитории (если пакет не найден локально, Go попытается загрузить его из интернета. Go использует URL, указанный в импорте, чтобы найти и загрузить исходный код пакета, затем сохранить его в локальном кэше)

### Параллелизм
1. **Каналы** -  позволяют горутинам общаться между собой. Это механизм синхронизации и обмена данными между горутинами. 
	- Буферизованный канал - имеет фиксированный размер и блокируется, когда он полон или пуст.
	- Небуферизованный канал - блокируется, пока одна горутина не отправит данные, а другая не получит их.
2. **select** — для работы с множественными каналами. Позволяет ждать событий на разных каналах (например, когда канал готов принять или вернуть данные) и обрабатывать их по мере поступления. Это похоже на оператор `switch`, но вместо условий проверяются каналы. Основные правила работы `select`:
	- `select` блокируется и ждет, пока одно из каналов станет доступным для операции (чтения или записи).
	- Если готовы несколько каналов одновременно, выбирается один случайным образом.
	- Если ни один из каналов не готов, `select` блокируется, пока не появится готовый канал.
	- Если в `select` есть случай с блоком `default`, он выполняется, если ни один канал не готов (что делает `select` неблокирующим).
3. **Mutex** - обычный мьютекс
4. **WaitGroup** - для синхронизации

### Выполнение программ

#### Как выполняются программы на Go?
1. *Компиляция*:
    - Исходный код Go компилируется в нативный машинный код для целевой платформы. Это означает, что программа становится автономным исполняемым файлом, который можно запустить без дополнительной среды исполнения.
    - Компилятор Go создает единственный статический бинарный файл, включающий все зависимости (стандартную библиотеку и прочее), что упрощает развертывание.
2. *Исполнение*:
    - После компиляции бинарный файл запускается непосредственно в операционной системе. Операционная система предоставляет базовые ресурсы, такие как процессорное время, память, доступ к сети и файловой системе.

#### Среда выполнения (runtime)
Хотя Go исполняется нативно, он включает в себя небольшую встроенную среду выполнения (runtime), которая предоставляет:
- *Управление горутинами*: планировщик, который управляет выполнением горутин.
- *Сборщик мусора*: освобождает память, от неиспользуемых объектов.
- *Обработку паник и восстановление* (`panic` и `recover`): Для работы с исключительными ситуациями.

#### Кроссплатформенность
Go поддерживает кросс-компиляцию, позволяя создавать бинарные файлы для разных операционных систем и архитектур, независимо от платформы разработки. Например: `GOOS=linux GOARCH=amd64 go build -o myprogram main.go` - Этот файл можно будет запустить на 64-битной Linux-системе.

### ООП

В Go нет привычной объектно-ориентированной модели программирования с классами, наследованием и другими традиционными механизмами ООП. Однако, язык предоставляет инструменты, которые позволяют реализовывать концепции объектно-ориентированного программирования.

Вместо классов - структуры. С помощью них можно определять новые типы данных:
```go
type Person struct {
	Name string
    Age  int
} 
```

Можно определять методы для структур:
```go
func (p Person) SayHello() {
	fmt.Println("Hello, my name is", p.Name)
}
```

В go нет наследования, но есть композиция. Поля или методы одного типа могут быть встроены в другой тип:
```go
type Employee struct {
	Person
	Position string
}
func main() {
	emp := Employee{
		Person: Person{Name: "Alice", Age: 30},
		Position: "Engineer", 
	}
	emp.SayHello()
}
```

Интерфейсы позволяют определить поведение и обеспечивают полиморфизм. Они определяют набор методов, которые должен реализовывать тип. Go не требуют явной привязки типа к интерфейсу — если тип реализует все методы интерфейса, он автоматически "удовлетворяет" интерфейсу:
```go
type Greeter interface {
	Greet()
}
func (p Person) Greet() {
	fmt.Println("Hi, I am", p.Name)
}
func SaySomething(g Greeter) {
	g.Greet()
}
func main() {
	p := Person{Name: "Bob"}
	SaySomething(p)
}
```

#### Философия Go
Хотя Go и не поддерживает традиционное ООП с классами, оно реализует *композиционное ООП*, которое делает упор на простоту и надёжность. Это позволяет писать модульный, легко поддерживаемый и эффективный код.

### Работа с памятью

Работа с памятью в Go организована так, чтобы быть максимально безопасной и автоматизированной, благодаря встроенному сборщику мусора.

**Указатель** — это переменная, которая хранит адрес другой переменной. Логика работы с указателями почти как в Си, но с некоторыми отличиями:
- *Отсутствие арифметики указателей*: В Go нельзя выполнять операции с указателями, такие как добавление или вычитание. Это сделано для повышения безопасности
- *Работа с nil*: Указатели по умолчанию имеют значение `nil`, если они не инициализированы. Причём `nil` нельзя разыменовать
- *Нет необходимости в явной работе с памятью*: В Go есть автоматический сборщик мусора, который освобождает неиспользуемую память, поэтому не нужно вручную освобождать память, как в C или C++
- *Разыменовывание указателя на структуру происходит автоматически*. То есть `p.X = 10` и `(*p).X = 10` это одно и то же (здесь p - указатель на структуру)

## Библиотеки
### Gio

#### Ссылки
- [Официальный Get Started](https://gioui.org/doc/learn/get-started)
- [Понятный Get Started](https://jonegil.github.io/gui-with-gio/)

#### Команды
- `go run gioui.org/example/hello@latest` - проверка, что всё установлено
- `go run -ldflags="-H windowsgui" gioui.org/example/hello@latest` - запуск без консоли

#### Основные понятия и компоненты Gio

**Декларативный подход**: Gio использует декларативную модель для построения интерфейсов. Это означает, что вы описываете, как должен выглядеть интерфейс, исходя из текущего состояния программы. Когда состояние изменяется, Gio пересчитывает интерфейс и обновляет только те элементы, которые изменились.

**События**:
- `Gio` поддерживает обработку событий пользователя, таких как клики мышью, нажатия клавиш и касания на сенсорных экранах.
- Для обработки событий используется специальный цикл, где события обрабатываются через метод `Events()` окна.
- События поступают в виде конкретных типов, таких как `app.UpdateEvent`, `app.DestroyEvent`, `widget.Clickable.Event()` и другие.

**Окно** — это основное место, где размещается интерфейс приложения.

**Контекст рисования** (`op.Ops`), с которым связано каждое окно, хранит команды для отрисовки интерфейса. Через контекст происходит обновление интерфейса в ответ на действия пользователя или изменения состояния.

**Виджеты** - элементы интерфейса (кнопки, текстовые поля, заголовки, слайдеры и т.д.)
Виджеты состоит из двух частей:
- Реальная - имеет состояние
- Графическая - обёртка над виджетом для отрисовки
**Constraints** (ограничения) - минимальный и максимальный размер виджета
**Dimensions** (размеры) - это фактический размер виджета
`widget` - пакет обеспечивающий базовые функциональные возможности компонентов пользовательского интерфейса (отслеживание состояния, обработка событий, наведена ли мышь на кнопку, была ли она нажата и т.д.)
`widget/material` - пакет предоставляющий тему

**layout** - это набор инструментов для управления расположением виджетов.

### Ebiten

#### Ссылки
- [Официальный Сайт](https://ebitengine.org)
- [Package Documentation](https://pkg.go.dev/github.com/hajimehoshi/ebiten/v2)
- [Cheat Sheet](https://ebitengine.org/en/documents/cheatsheet.html)
- [Видео гайд по созданию сетевой игры](https://www.youtube.com/watch?v=jMqC_VUEAgs)

#### Layout

Функция `Layout` используется для установки фиксированного размера виртуального экрана (отображаемой области), вне зависимости от фактических размеров окна на экране. Это полезно, так как позволяет масштабировать контент игры на разные разрешения дисплея, сохраняя исходные пропорции.

`func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int)`
1. Параметры:
    - `outsideWidth` и `outsideHeight` — фактическая ширина и высота окна на экране.
    - Эти параметры Ebiten передает каждый раз, когда окно изменяет размер (например, пользователь растягивает окно).
2. Возвращаемые значения*:
    - `screenWidth` и `screenHeight` — ширина и высота виртуального экрана, которые будут использоваться для отрисовки. Эти значения показывают, что надо отобразить (какую часть игрового мира).
    - Вы можете установить их как фиксированные значения (например, 320x240), и Ebiten масштабирует всё внутри окна, чтобы соответствовать этим размерам.

### Proto

Protocol Buffers (protobuf) используется для сериализации и десериализации структурированных данных, что позволяет эффективно обмениваться данными между различными сервисами или хранить их в файловой системе. 

#### Использование
- [Официальный репозиторий](https://github.com/protocolbuffers/protobuf/releases) - здесь скачиваем компилятор protoc и устанавливаем соответствующие переменные среды
- `.proto` - в этом файле описываем все нужные структур данных
- `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest` - скачиваем плагин, чтобы protoc знал, как работать с go
- `protoc --go_out=. snakes.proto` - генерация кода на go со всеми необходимые структурами и методами для работы с вашими сообщениями
- Далее в вашем коде вы можете использовать сгенерированные структуры для создания объектов, их сериализации в бинарный формат и десериализации обратно.
- После сериализации вы можете отправить данные по сети, сохранить в файл или использовать по своему усмотрению. На принимающей стороне вы можете десериализовать данные, чтобы получить доступ к исходной структуре.

#### Преимущества использования Protocol Buffers
- **Эффективность**: Протокол Buffers сериализует данные в компактный бинарный формат, что экономит место и уменьшает время передачи.
- **Совместимость**: Поддержка версионирования позволяет добавлять новые поля и изменять структуру данных без нарушения обратной совместимости.
- **Многоязычность**: Генерация кода для разных языков позволяет использовать protobuf в различных средах и языках программирования.
