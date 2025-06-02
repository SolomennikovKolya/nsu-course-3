module Langs;

var $min_diff, $max_diff : integer;          // Выбранные границы сложности
var $min_pop, $max_pop : integer;            // Выбранные границы популярности
var $Win_lang, $Win_desc : integer;          // Окна для подходящих языков и выбранных критериев
var $y_lang, $y_desc : integer;              // Координаты для печати текста в этих окнах
var $descMenu : set of string;               // Множество критериев для выбора в меню
var $descChoise : set of string;             // Множество выбранных критериев
var $tupleMenu : tuple of string;            // Преобразованный $descMenu
var $choice : string;                        // Только что сделаный выбор
var $langs : integer;                        // Кол-во подходящих языков
var $t: tuple of string := string[];         // Временный tuple of string
var $flag : integer;                         // Флаг для выхода из цикла
var $selected_n : integer;                   // Номер выбранного варианта из меню
var $allLangs : tuple of string := string[]; // Все языки
var $selectedLang : string;                  // Выбранный язык
var $Win_info, $y_info : integer;            // Окни для вывода информации о языке и координата

// Преобразование set в tuple
function SetToTuple($S: set of string): tuple of string
begin
  $t := string[];
  for $e in $S loop
    $t := $t + string[$e];
  end;
  return $t;
end;

// Правило для остановки
rule Stop => activate group(); end;

// Вывод подходящих языков программирования и обновление вариантов для выбора в меню 
rule Print_Langs
  forall $l: lang(name: $name, features: $f, difficulty: $d, popularity: $pop) when
    ($d >= $min_diff) & ($d <= $max_diff) &
    ($pop >= $min_pop) & ($pop <= $max_pop) &
    ($descChoise <= $f)
=>
  $langs := $langs + 1;
  OutText($Win_lang, 10, $y_lang, $name);
  $y_lang := $y_lang + 15;

  $descMenu := $descMenu + $f - $descChoise;
end;

// Выбор нового критерия
rule New_Descs =>
  $descChoise := $descChoise + string{$choice};
  OutText($Win_desc, 10, $y_desc, $choice);
  $y_desc := $y_desc + 15;
end;

// Получение списка всех языков
rule GetAllLangs
  forall $l: lang(name: $n)
=>
  $allLangs := $allLangs + string[$n];
end;

// Вывод информации о языке
rule PrintLangInfo
  forall $l: lang(name: $name, features: $f, difficulty: $d, popularity: $p) when $name = $selectedLang
=>
  OutText($Win_info, 10, $y_info, "Название: " + $selectedLang); $y_info := $y_info + 15;
  OutText($Win_info, 10, $y_info, "Сложность: " + ToString($d)); $y_info := $y_info + 15;
  OutText($Win_info, 10, $y_info, "Популярность: " + ToString($p)); $y_info := $y_info + 15;
  OutText($Win_info, 10, $y_info, "Критерии:"); $y_info := $y_info + 15;

  for $feat in $f loop
      OutText($Win_info, 20, $y_info, "- " + $feat);
      $y_info := $y_info + 15;
  end;
end;

begin
  new
    @l1: lang(name: "Python", features: string{
      "универсальный", "анализ данных", "веб", "скрипты", "настольный", "серверный", "кроссплатформенный",
      "императивный", "ООП", "интерпретируемый", "динамическая типизация", "сильная типизация", "неявная типизация",
      "сборщик мусора", "менеджер пакетов", "официальный инструментарий"
    }, difficulty: 3, popularity: 95),

    @l2: lang(name: "C++", features: string{
      "универсальный", "игры", "системное", "нативный", "настольный", "серверный",
      "императивный", "ООП", "процедурный", "компилируемый", "статическая типизация", "слабая типизация", "явная типизация",
      "ручное управление памятью", "обобщения"
    }, difficulty: 8, popularity: 85),

    @l3: lang(name: "Java", features: string{
      "универсальный", "веб", "мобильная разработка", "настольный", "серверный", "кроссплатформенный",
      "императивный", "ООП", "байткод", "статическая типизация", "сильная типизация", "явная типизация",
      "сборщик мусора", "менеджер пакетов", "официальный инструментарий", "обобщения"
    }, difficulty: 5, popularity: 88),

    @l4: lang(name: "JavaScript", features: string{
      "веб", "браузер", "универсальный", "императивный", "функциональный",
      "интерпретируемый", "динамическая типизация", "слабая типизация", "неявная типизация",
      "сборщик мусора", "менеджер пакетов"
    }, difficulty: 4, popularity: 92),

    @l5: lang(name: "Go", features: string{
      "универсальный", "веб", "серверный", "системное",
      "компилируемый", "императивный", "статическая типизация", "сильная типизация", "явная типизация",
      "сборщик мусора", "официальный инструментарий"
    }, difficulty: 4, popularity: 75),

    @l6: lang(name: "Rust", features: string{
      "системное", "игры", "нативный", "серверный",
      "компилируемый", "императивный", "функциональный", "статическая типизация", "сильная типизация", "явная типизация",
      "ручное управление памятью", "официальный инструментарий", "обобщения"
    }, difficulty: 9, popularity: 70),

    @l7: lang(name: "Haskell", features: string{
      "академический", "функциональный", "декларативный",
      "компилируемый", "статическая типизация", "сильная типизация", "явная типизация",
      "менеджер пакетов"
    }, difficulty: 9, popularity: 40),

    @l8: lang(name: "Swift", features: string{
      "мобильная разработка", "настольный", "универсальный",
      "императивный", "ООП", "компилируемый", "статическая типизация", "сильная типизация", "неявная типизация",
      "сборщик мусора", "официальный инструментарий"
    }, difficulty: 5, popularity: 65),

    @l9: lang(name: "Kotlin", features: string{
      "мобильная разработка", "веб", "серверный", "универсальный",
      "ООП", "императивный", "байткод", "статическая типизация", "сильная типизация", "неявная типизация",
      "сборщик мусора", "менеджер пакетов", "официальный инструментарий", "обобщения"
    }, difficulty: 4, popularity: 60),

    @l10: lang(name: "TypeScript", features: string{
      "веб", "браузер", "универсальный",
      "ООП", "функциональный", "транспилируемый", "статическая типизация", "сильная типизация", "неявная типизация",
      "менеджер пакетов"
    }, difficulty: 4, popularity: 70),

    @l11: lang(name: "C#", features: string{
      "универсальный", "веб", "настольный", "игры",
      "ООП", "компилируемый", "байткод", "статическая типизация", "сильная типизация", "явная типизация",
      "сборщик мусора", "официальный инструментарий", "обобщения"
    }, difficulty: 5, popularity: 80),

    @l12: lang(name: "Ruby", features: string{
      "веб", "скрипты", "интерпретируемый", "динамическая типизация", "сильная типизация", "неявная типизация",
      "ООП", "сборщик мусора", "менеджер пакетов"
    }, difficulty: 4, popularity: 60),

    @l13: lang(name: "PHP", features: string{
      "веб", "серверный", "интерпретируемый", "динамическая типизация", "слабая типизация", "неявная типизация",
      "императивный", "менеджер пакетов"
    }, difficulty: 3, popularity: 65),

    @l14: lang(name: "Perl", features: string{
      "скрипты", "универсальный", "интерпретируемый", "динамическая типизация", "слабая типизация", "неявная типизация"
    }, difficulty: 5, popularity: 40),

    @l15: lang(name: "Scala", features: string{
      "универсальный", "веб", "серверный",
      "функциональный", "ООП", "байткод", "статическая типизация", "сильная типизация", "неявная типизация",
      "обобщения", "сборщик мусора"
    }, difficulty: 6, popularity: 55),

    @l16: lang(name: "Elixir", features: string{
      "веб", "серверный", "функциональный", "интерпретируемый", "динамическая типизация", "сильная типизация", "неявная типизация"
    }, difficulty: 6, popularity: 45),

    @l17: lang(name: "F#", features: string{
      "академический", "функциональный", "ООП", "компилируемый", "статическая типизация", "сильная типизация", "неявная типизация"
    }, difficulty: 7, popularity: 35),

    @l18: lang(name: "Lua", features: string{
      "игры", "встраиваемый", "интерпретируемый", "динамическая типизация", "слабая типизация", "неявная типизация"
    }, difficulty: 3, popularity: 50),

    @l19: lang(name: "Assembly", features: string{
      "системное", "нативный", "настольный", "ручное управление памятью", "компилируемый"
    }, difficulty: 10, popularity: 30),

    @l20: lang(name: "R", features: string{
      "анализ данных", "академический", "интерпретируемый", "динамическая типизация", "неявная типизация",
      "менеджер пакетов"
    }, difficulty: 4, popularity: 55),

    @l21: lang(name: "MATLAB", features: string{
      "академический", "анализ данных", "интерпретируемый", "динамическая типизация", "неявная типизация"
    }, difficulty: 5, popularity: 50),

    @l22: lang(name: "Bash", features: string{
      "скрипты", "интерпретируемый", "динамическая типизация", "неявная типизация"
    }, difficulty: 3, popularity: 65),

    @l23: lang(name: "Objective-C", features: string{
      "мобильная разработка", "настольный", "ООП", "компилируемый", "статическая типизация", "явная типизация"
    }, difficulty: 6, popularity: 40),

    @l24: lang(name: "Dart", features: string{
      "веб", "мобильная разработка", "браузер", "OOП", "императивный",
      "байткод", "статическая типизация", "неявная типизация", "сборщик мусора"
    }, difficulty: 4, popularity: 60),

    @l25: lang(name: "Julia", features: string{
      "анализ данных", "академический", "высокая производительность", "динамическая типизация", "неявная типизация",
      "функциональный", "интерпретируемый", "менеджер пакетов"
    }, difficulty: 5, popularity: 50);

  if Ask("", "Хотите ли вы просмотреть информацию о языке?") then
    call group(GetAllLangs, Stop);

    $selected_n := Menu(20, 100, "Выберите язык", $allLangs, 0);
    if $selected_n != 0 then
      $selectedLang := $allLangs[$selected_n];
      $Win_info := MakeWindow("Информация о " + $selectedLang, 100, 100, 450, 400);
      TextColor($Win_info, 4);
      $y_info := 10;
      call group(PrintLangInfo, Stop);
      Message("", "Нажмите OK для завершения.");
      CloseWindow($Win_info);
    end;

  else
    $descMenu := string{};
    $descChoise := string{};

    // Создание окон
    $Win_lang := MakeWindow("Подходящие языки", 450, 20, 300, 430);
    $Win_desc := MakeWindow("Выбранные критерии", 20, 20, 420, 170);
    TextColor($Win_lang, 3);
    TextColor($Win_desc, 1);
    $y_lang := 10;
    $y_desc := 10;

    // Выбор сложности
    $min_diff := GetNumber(100, 100, "Сложность", "Минимум от 1 до 10:", 1);
    $max_diff := GetNumber(100, 100, "Сложность", "Максимум от 1 до 10:", 10);
    OutText($Win_desc, 10, $y_desc, "Сложность: " + ToString($min_diff) + "–" + ToString($max_diff));
    $y_desc := $y_desc + 15;

    // Выбор популярности
    $min_pop := GetNumber(100, 100, "Популярность", "Минимум от 1 до 100:", 1);
    $max_pop := GetNumber(100, 100, "Популярность", "Максимум от 1 до 100:", 100);
    OutText($Win_desc, 10, $y_desc, "Популярность: " + ToString($min_pop) + "–" + ToString($max_pop));
    $y_desc := $y_desc + 15;

    $langs := 0;
    call group(Print_Langs, Stop);

    $flag := 1;
    while ($flag != 0 & $langs > 1) loop
      $tupleMenu := SetToTuple($descMenu);
      $selected_n := Menu(20, 200, "Выберите критерии", $tupleMenu, 0);

      if $selected_n != 0 then
        $choice := $tupleMenu[$selected_n];
        WriteLn($tupleMenu);
        WriteLn($choice);

        ClearWindow($Win_lang);
        $y_lang := 10;
        $descMenu := string{};
        $langs := 0;
        call group(New_Descs, Print_Langs, Stop);
        if #$descMenu = 0 then $flag := 0; end;
      end;
    end;

    Message("", "Выбор завершён");
    CloseWindow($Win_lang);
    CloseWindow($Win_desc);
  end;
end.
