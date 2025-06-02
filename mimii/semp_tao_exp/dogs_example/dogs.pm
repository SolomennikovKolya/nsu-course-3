module Dogs;

var $s_min, $s_max, $w_min, $w_max, $y_breed, $y_desc, $dogs : integer;
var $Win_breed, $Win_descs : integer;
var $descMenu, $descChoise : set of string;
var $d_Menu : tuple of string;
var $ch_desc : string;

// Преобразование set в tuple
function SetToTuple($S: set of string): tuple of string
begin
var $t: tuple of string := string[];
  for $e in $S loop
   $t:=$t+string[$e];
  end;
  return $t;
end;

// Правило для остановки
rule Stop
=>
  activate group();
end;

// Выводит список выбранных пород
rule Print_Breeds
  forall $d: dog(breed: $breed, sizemin: $smin, sizemax: $smax, weightmin: $wmin, weightmax: $wmax, description: $desc) when
    ($smax <= $s_max) & ($smin >= $s_min) & ($wmax <= $w_max) & ($wmin >= $w_min) & ($descChoise <= $desc)
=>
  var $s : set of string;
  var $flag : integer;

  $dogs:=$dogs + 1; 
  OutText($Win_breed, 10, $y_breed, $breed);
  $y_breed := $y_breed + 15;
  
  $descMenu:=$descMenu+$desc-$descChoise;
end;

// Изменение множества descMenu и вывод выборки
rule New_Descs
=>
  $descChoise:=$descChoise+string{$ch_desc}; 
  OutText($Win_descs,10,$y_desc,$ch_desc);
  $y_desc:=$y_desc+15;
end;

begin
  var $n : integer;
  new 
    @dog1  : dog(breed: "Афганская борзая", sizemin: 61, sizemax: 71,weightmin: 23, weightmax: 28, description: string{"борзая","охотник на крупного зверя","требует регулярных физических нагрузок","требуется специальный уход за шерстью"}),  
    @dog2  : dog(breed: "Американский бульдог", sizemin: 45, sizemax: 64,weightmin:30, weightmax:48, description: string{"идеальный домашний любимец","несложный уход за шерстью","может быть обучен задержанию преступников"}),  
    @dog3  : dog(breed: "Боксер", sizemin: 57, sizemax: 64,weightmin:24, weightmax:28, description: string{"идеальный домашний любимец","несложный уход за шерстью","пригодна для караульной и сторожевой службы","может быть проводником слепых"}),  
    @dog4  : dog(breed: "Доберман-пинчер", sizemin: 59, sizemax: 71,weightmin:30, weightmax:40, description: string{"идеальный домашний любимец","несложный уход за шерстью","может быть проводником слепых","хорошо обучается для разыскной и спасательной службы","может быть обучен задержанию преступников"}),  
    @dog5  : dog(breed: "Немецкий дог", sizemin: 71, sizemax: 80,weightmin: 45, weightmax:90, description: string{"идеальный домашний любимец","охотник на крупного зверя","требует регулярных физических нагрузок","пригодна для караульной и сторожевой службы"}),  
    @dog6  : dog(breed: "Ньюфаундленд", sizemin: 66, sizemax: 76,weightmin: 50, weightmax:58, description: string{"идеальный домашний любимец","хорошо обучается для разыскной и спасательной службы","собака-пловец","шерстный покров защищает в любую погоду","собака, подающая подстреленную дичь"}),  
    @dog7  : dog(breed: "Сенбернар", sizemin: 65, sizemax: 70,weightmin:50, weightmax:70, description: string{"идеальный домашний любимец","хорошо обучается для разыскной и спасательной службы","собака-пловец","шерстный покров защищает в любую погоду"}),  
    @dog8  : dog(breed: "Чау-чау", sizemin: 48, sizemax: 51,weightmin:21, weightmax:32, description: string{"идеальный домашний любимец","пригодна для караульной и сторожевой службы","требуется специальный уход за шерстью","охотник на мелкую дичь","северная ездовая собака"}),  
    @dog9  : dog(breed: "Американский кокер-спаниель", sizemin: 35, sizemax: 38,weightmin:11, weightmax:13, description: string{"идеальный домашний любимец","требуется специальный уход за шерстью","охотник на мелкую дичь","подружейная собака"}),  
    @dog10 : dog(breed: "Миниатюрный пудель", sizemin: 25, sizemax: 38,weightmin: 5, weightmax:8, description: string{"идеальный домашний любимец","хорошо приспособлен для жизни в квартире","подружейная собака","требуется специальный уход за шерстью"}),  
    @dog11 : dog(breed: "Стандартный пудель", sizemin: 38, sizemax: 45,weightmin:21, weightmax:34, description: string{"идеальный домашний любимец","требуется специальный уход за шерстью","собака-пловец","гончая","требует регулярных физических нагрузок"}),  
    @dog12 : dog(breed: "Бассет-хаунд", sizemin: 35, sizemax: 36,weightmin:18, weightmax:27, description: string{"идеальный домашний любимец","охотник на мелкую дичь","несложный уход за шерстью","требует регулярных физических нагрузок","гончая"}),  
    @dog13 : dog(breed: "Миниатюрный пинчер", sizemin: 25, sizemax: 32,weightmin:3, weightmax:4, description: string{"идеальный домашний любимец","несложный уход за шерстью","терьер-крысолов","декоративная собака"}),  
    @dog14 : dog(breed: "Пекинес", sizemin: 20, sizemax: 22,weightmin:2, weightmax:5, description: string{"декоративная собака","хорошо приспособлен для жизни в квартире","пригодна для караульной и сторожевой службы","требуется специальный уход за шерстью"}),  
    @dog15 : dog(breed: "Кавказская овчарка", sizemin: 63, sizemax: 71,weightmin:46, weightmax:66, description: string{"требует регулярных физических нагрузок","шерстный покров защищает в любую погоду","может быть обучен задержанию преступников","агрессивная собака","пастушья сторожевая собака"}),
    @dog16 : dog(breed: "Русская псовая борзая", sizemin: 71, sizemax: 79, weightmin: 34, weightmax: 48, description: string{"борзая","идеальный домашний любимец","охотник на крупного зверя","шерстный покров защищает в любую погоду"}),
    @dog17 : dog(breed: "Английская борзая", sizemin: 68, sizemax: 76, weightmin: 27, weightmax: 32, description: string{"борзая","идеальный домашний любимец","требует регулярных физических нагрузок","борзая для беговых состязаний"}),
    @dog18 : dog(breed: "Среднеазиатская борзая", sizemin: 56, sizemax: 71, weightmin: 25, weightmax: 30, description: string{"борзая","идеальный домашний любимец","охотник на крупного зверя","шерстный покров защищает в любую погоду","охотник на мелкую дичь"}),
    @dog19 : dog(breed: "Бультерьер", sizemin: 53, sizemax: 56, weightmin: 24, weightmax: 28, description: string{"несложный уход за шерстью","терьер-крысолов","пригодна для караульной и сторожевой службы"}),
    @dog20 : dog(breed: "Ландсир", sizemin: 66, sizemax: 80, weightmin: 50, weightmax: 58, description: string{"идеальный домашний любимец","шерстный покров защищает в любую погоду","собака-пловец","хорошо обучается для разыскной и спасательной службы","собака, подающая подстреленную дичь"}),
    @dog21 : dog(breed: "Английский мастиф", sizemin: 70, sizemax: 76, weightmin: 80, weightmax: 86, description: string{"несложный уход за шерстью","требует регулярных физических нагрузок","пастушья сторожевая собака","пригодна для караульной и сторожевой службы"}),
    @dog22 : dog(breed: "Московская сторожевая", sizemin: 64, sizemax: 69, weightmin: 45, weightmax: 68, description: string{"идеальный домашний любимец","требует регулярных физических нагрузок","может быть проводником слепых","пригодна для караульной и сторожевой службы"}),
    @dog23 : dog(breed: "Ротвейлер", sizemin: 56, sizemax: 69, weightmin: 41, weightmax: 50, description: string{"пастух стад крупного рогатого скота","несложный уход за шерстью","требует регулярных физических нагрузок","может быть обучен задержанию преступников","пригодна для караульной и сторожевой службы"}),
    @dog24 : dog(breed: "Самоедская лайка", sizemin: 46, sizemax: 56, weightmin: 23, weightmax: 30, description: string{"идеальный домашний любимец","шерстный покров защищает в любую погоду","пригодна для караульной и сторожевой службы","требуется специальный уход за шерстью","пастух овечьих отар"});

  $descMenu  := string{}; 
  $descChoise:= string{};

  // Создание окна для вывода выбранных собак
  $Win_breed := MakeWindow("Подходящие породы собак", 450, 20, 300, 430);
  TextColor($Win_breed, 3);
  $y_breed := 10;

  // Создание окна для вывода выбранных характеристик
  $Win_descs := MakeWindow("Выбранные характеристики собак", 20, 20, 420, 170);
  TextColor($Win_descs, 1);
  $y_desc := 10;

  // Начальная выборка по высоте и весу 
  if Ask("", "Делать выбор по росту собаки?") then
    $s_min := GetNumber(100, 100, "Высота собаки в холке (в см)", "Введите минимальную высоту:", 1);
    $s_max := GetNumber(100, 100, "Высота собаки в холке (в см)", "Введите максимальную высоту:", 100);
    OutText($Win_descs,10,$y_desc,"Рост от "+ToString($s_min)+" до "+ToString($s_max)+" см");
    $y_desc:=$y_desc+15;
  else
    $s_min := 1; 
    $s_max := 100;
  end;
  if Ask("", "Делать выбор по весу собаки?") then
    $w_min := GetNumber(100, 100, "Вес собаки (в кг)", "Введите минимальный вес:", 1);
    $w_max := GetNumber(100, 100, "Вес собаки (в кг)", "Введите максимальный вес:", 100);
    OutText($Win_descs,10,$y_desc,"Вес от "+ToString($w_min)+" до "+ToString($w_max)+" кг");
    $y_desc:=$y_desc+15;
  else
    $w_min := 1; 
    $w_max := 100;
  end;

  $dogs:=0;
  call group(Print_Breeds,Stop);
  $n:=1;
  while ($n!=0 & $dogs>1) loop
    $d_Menu:=SetToTuple($descMenu);
    $n := Menu(20, 200, "Выберите характеристики собаки", $d_Menu, 0);
    if $n!=0 then
      WriteLn($d_Menu);
      $ch_desc:=$d_Menu[$n];
      WriteLn($ch_desc);
      ClearWindow($Win_breed);
      $y_breed:=10;
      $descMenu:=string{}; 
      $dogs:=0;
      call group(New_Descs,Print_Breeds,Stop);
      if #$descMenu=0 then $n:=0; end;
    end; 
  end;
  Message("", "Выбор закончен");
  CloseWindow($Win_breed);
  CloseWindow($Win_descs);
end.