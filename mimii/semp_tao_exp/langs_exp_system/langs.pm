module Langs;

var $min_diff, $max_diff : integer;          // ��������� ������� ���������
var $min_pop, $max_pop : integer;            // ��������� ������� ������������
var $Win_lang, $Win_desc : integer;          // ���� ��� ���������� ������ � ��������� ���������
var $y_lang, $y_desc : integer;              // ���������� ��� ������ ������ � ���� �����
var $descMenu : set of string;               // ��������� ��������� ��� ������ � ����
var $descChoise : set of string;             // ��������� ��������� ���������
var $tupleMenu : tuple of string;            // ��������������� $descMenu
var $choice : string;                        // ������ ��� �������� �����
var $langs : integer;                        // ���-�� ���������� ������
var $t: tuple of string := string[];         // ��������� tuple of string
var $flag : integer;                         // ���� ��� ������ �� �����
var $selected_n : integer;                   // ����� ���������� �������� �� ����
var $allLangs : tuple of string := string[]; // ��� �����
var $selectedLang : string;                  // ��������� ����
var $Win_info, $y_info : integer;            // ���� ��� ������ ���������� � ����� � ����������

// �������������� set � tuple
function SetToTuple($S: set of string): tuple of string
begin
  $t := string[];
  for $e in $S loop
    $t := $t + string[$e];
  end;
  return $t;
end;

// ������� ��� ���������
rule Stop => activate group(); end;

// ����� ���������� ������ ���������������� � ���������� ��������� ��� ������ � ���� 
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

// ����� ������ ��������
rule New_Descs =>
  $descChoise := $descChoise + string{$choice};
  OutText($Win_desc, 10, $y_desc, $choice);
  $y_desc := $y_desc + 15;
end;

// ��������� ������ ���� ������
rule GetAllLangs
  forall $l: lang(name: $n)
=>
  $allLangs := $allLangs + string[$n];
end;

// ����� ���������� � �����
rule PrintLangInfo
  forall $l: lang(name: $name, features: $f, difficulty: $d, popularity: $p) when $name = $selectedLang
=>
  OutText($Win_info, 10, $y_info, "��������: " + $selectedLang); $y_info := $y_info + 15;
  OutText($Win_info, 10, $y_info, "���������: " + ToString($d)); $y_info := $y_info + 15;
  OutText($Win_info, 10, $y_info, "������������: " + ToString($p)); $y_info := $y_info + 15;
  OutText($Win_info, 10, $y_info, "��������:"); $y_info := $y_info + 15;

  for $feat in $f loop
      OutText($Win_info, 20, $y_info, "- " + $feat);
      $y_info := $y_info + 15;
  end;
end;

begin
  new
    @l1: lang(name: "Python", features: string{
      "�������������", "������ ������", "���", "�������", "����������", "���������", "������������������",
      "������������", "���", "����������������", "������������ ���������", "������� ���������", "������� ���������",
      "������� ������", "�������� �������", "����������� ��������������"
    }, difficulty: 3, popularity: 95),

    @l2: lang(name: "C++", features: string{
      "�������������", "����", "���������", "��������", "����������", "���������",
      "������������", "���", "�����������", "�������������", "����������� ���������", "������ ���������", "����� ���������",
      "������ ���������� �������", "���������"
    }, difficulty: 8, popularity: 85),

    @l3: lang(name: "Java", features: string{
      "�������������", "���", "��������� ����������", "����������", "���������", "������������������",
      "������������", "���", "�������", "����������� ���������", "������� ���������", "����� ���������",
      "������� ������", "�������� �������", "����������� ��������������", "���������"
    }, difficulty: 5, popularity: 88),

    @l4: lang(name: "JavaScript", features: string{
      "���", "�������", "�������������", "������������", "��������������",
      "����������������", "������������ ���������", "������ ���������", "������� ���������",
      "������� ������", "�������� �������"
    }, difficulty: 4, popularity: 92),

    @l5: lang(name: "Go", features: string{
      "�������������", "���", "���������", "���������",
      "�������������", "������������", "����������� ���������", "������� ���������", "����� ���������",
      "������� ������", "����������� ��������������"
    }, difficulty: 4, popularity: 75),

    @l6: lang(name: "Rust", features: string{
      "���������", "����", "��������", "���������",
      "�������������", "������������", "��������������", "����������� ���������", "������� ���������", "����� ���������",
      "������ ���������� �������", "����������� ��������������", "���������"
    }, difficulty: 9, popularity: 70),

    @l7: lang(name: "Haskell", features: string{
      "�������������", "��������������", "�������������",
      "�������������", "����������� ���������", "������� ���������", "����� ���������",
      "�������� �������"
    }, difficulty: 9, popularity: 40),

    @l8: lang(name: "Swift", features: string{
      "��������� ����������", "����������", "�������������",
      "������������", "���", "�������������", "����������� ���������", "������� ���������", "������� ���������",
      "������� ������", "����������� ��������������"
    }, difficulty: 5, popularity: 65),

    @l9: lang(name: "Kotlin", features: string{
      "��������� ����������", "���", "���������", "�������������",
      "���", "������������", "�������", "����������� ���������", "������� ���������", "������� ���������",
      "������� ������", "�������� �������", "����������� ��������������", "���������"
    }, difficulty: 4, popularity: 60),

    @l10: lang(name: "TypeScript", features: string{
      "���", "�������", "�������������",
      "���", "��������������", "���������������", "����������� ���������", "������� ���������", "������� ���������",
      "�������� �������"
    }, difficulty: 4, popularity: 70),

    @l11: lang(name: "C#", features: string{
      "�������������", "���", "����������", "����",
      "���", "�������������", "�������", "����������� ���������", "������� ���������", "����� ���������",
      "������� ������", "����������� ��������������", "���������"
    }, difficulty: 5, popularity: 80),

    @l12: lang(name: "Ruby", features: string{
      "���", "�������", "����������������", "������������ ���������", "������� ���������", "������� ���������",
      "���", "������� ������", "�������� �������"
    }, difficulty: 4, popularity: 60),

    @l13: lang(name: "PHP", features: string{
      "���", "���������", "����������������", "������������ ���������", "������ ���������", "������� ���������",
      "������������", "�������� �������"
    }, difficulty: 3, popularity: 65),

    @l14: lang(name: "Perl", features: string{
      "�������", "�������������", "����������������", "������������ ���������", "������ ���������", "������� ���������"
    }, difficulty: 5, popularity: 40),

    @l15: lang(name: "Scala", features: string{
      "�������������", "���", "���������",
      "��������������", "���", "�������", "����������� ���������", "������� ���������", "������� ���������",
      "���������", "������� ������"
    }, difficulty: 6, popularity: 55),

    @l16: lang(name: "Elixir", features: string{
      "���", "���������", "��������������", "����������������", "������������ ���������", "������� ���������", "������� ���������"
    }, difficulty: 6, popularity: 45),

    @l17: lang(name: "F#", features: string{
      "�������������", "��������������", "���", "�������������", "����������� ���������", "������� ���������", "������� ���������"
    }, difficulty: 7, popularity: 35),

    @l18: lang(name: "Lua", features: string{
      "����", "������������", "����������������", "������������ ���������", "������ ���������", "������� ���������"
    }, difficulty: 3, popularity: 50),

    @l19: lang(name: "Assembly", features: string{
      "���������", "��������", "����������", "������ ���������� �������", "�������������"
    }, difficulty: 10, popularity: 30),

    @l20: lang(name: "R", features: string{
      "������ ������", "�������������", "����������������", "������������ ���������", "������� ���������",
      "�������� �������"
    }, difficulty: 4, popularity: 55),

    @l21: lang(name: "MATLAB", features: string{
      "�������������", "������ ������", "����������������", "������������ ���������", "������� ���������"
    }, difficulty: 5, popularity: 50),

    @l22: lang(name: "Bash", features: string{
      "�������", "����������������", "������������ ���������", "������� ���������"
    }, difficulty: 3, popularity: 65),

    @l23: lang(name: "Objective-C", features: string{
      "��������� ����������", "����������", "���", "�������������", "����������� ���������", "����� ���������"
    }, difficulty: 6, popularity: 40),

    @l24: lang(name: "Dart", features: string{
      "���", "��������� ����������", "�������", "OO�", "������������",
      "�������", "����������� ���������", "������� ���������", "������� ������"
    }, difficulty: 4, popularity: 60),

    @l25: lang(name: "Julia", features: string{
      "������ ������", "�������������", "������� ������������������", "������������ ���������", "������� ���������",
      "��������������", "����������������", "�������� �������"
    }, difficulty: 5, popularity: 50);

  if Ask("", "������ �� �� ����������� ���������� � �����?") then
    call group(GetAllLangs, Stop);

    $selected_n := Menu(20, 100, "�������� ����", $allLangs, 0);
    if $selected_n != 0 then
      $selectedLang := $allLangs[$selected_n];
      $Win_info := MakeWindow("���������� � " + $selectedLang, 100, 100, 450, 400);
      TextColor($Win_info, 4);
      $y_info := 10;
      call group(PrintLangInfo, Stop);
      Message("", "������� OK ��� ����������.");
      CloseWindow($Win_info);
    end;

  else
    $descMenu := string{};
    $descChoise := string{};

    // �������� ����
    $Win_lang := MakeWindow("���������� �����", 450, 20, 300, 430);
    $Win_desc := MakeWindow("��������� ��������", 20, 20, 420, 170);
    TextColor($Win_lang, 3);
    TextColor($Win_desc, 1);
    $y_lang := 10;
    $y_desc := 10;

    // ����� ���������
    $min_diff := GetNumber(100, 100, "���������", "������� �� 1 �� 10:", 1);
    $max_diff := GetNumber(100, 100, "���������", "�������� �� 1 �� 10:", 10);
    OutText($Win_desc, 10, $y_desc, "���������: " + ToString($min_diff) + "�" + ToString($max_diff));
    $y_desc := $y_desc + 15;

    // ����� ������������
    $min_pop := GetNumber(100, 100, "������������", "������� �� 1 �� 100:", 1);
    $max_pop := GetNumber(100, 100, "������������", "�������� �� 1 �� 100:", 100);
    OutText($Win_desc, 10, $y_desc, "������������: " + ToString($min_pop) + "�" + ToString($max_pop));
    $y_desc := $y_desc + 15;

    $langs := 0;
    call group(Print_Langs, Stop);

    $flag := 1;
    while ($flag != 0 & $langs > 1) loop
      $tupleMenu := SetToTuple($descMenu);
      $selected_n := Menu(20, 200, "�������� ��������", $tupleMenu, 0);

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

    Message("", "����� ��������");
    CloseWindow($Win_lang);
    CloseWindow($Win_desc);
  end;
end.
