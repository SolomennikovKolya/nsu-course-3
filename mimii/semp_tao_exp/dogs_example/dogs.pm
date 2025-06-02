module Dogs;

var $s_min, $s_max, $w_min, $w_max, $y_breed, $y_desc, $dogs : integer;
var $Win_breed, $Win_descs : integer;
var $descMenu, $descChoise : set of string;
var $d_Menu : tuple of string;
var $ch_desc : string;

// �������������� set � tuple
function SetToTuple($S: set of string): tuple of string
begin
var $t: tuple of string := string[];
  for $e in $S loop
   $t:=$t+string[$e];
  end;
  return $t;
end;

// ������� ��� ���������
rule Stop
=>
  activate group();
end;

// ������� ������ ��������� �����
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

// ��������� ��������� descMenu � ����� �������
rule New_Descs
=>
  $descChoise:=$descChoise+string{$ch_desc}; 
  OutText($Win_descs,10,$y_desc,$ch_desc);
  $y_desc:=$y_desc+15;
end;

begin
  var $n : integer;
  new 
    @dog1  : dog(breed: "��������� ������", sizemin: 61, sizemax: 71,weightmin: 23, weightmax: 28, description: string{"������","������� �� �������� �����","������� ���������� ���������� ��������","��������� ����������� ���� �� �������"}),  
    @dog2  : dog(breed: "������������ �������", sizemin: 45, sizemax: 64,weightmin:30, weightmax:48, description: string{"��������� �������� �������","��������� ���� �� �������","����� ���� ������ ���������� ������������"}),  
    @dog3  : dog(breed: "������", sizemin: 57, sizemax: 64,weightmin:24, weightmax:28, description: string{"��������� �������� �������","��������� ���� �� �������","�������� ��� ���������� � ���������� ������","����� ���� ����������� ������"}),  
    @dog4  : dog(breed: "��������-������", sizemin: 59, sizemax: 71,weightmin:30, weightmax:40, description: string{"��������� �������� �������","��������� ���� �� �������","����� ���� ����������� ������","������ ��������� ��� ��������� � ������������ ������","����� ���� ������ ���������� ������������"}),  
    @dog5  : dog(breed: "�������� ���", sizemin: 71, sizemax: 80,weightmin: 45, weightmax:90, description: string{"��������� �������� �������","������� �� �������� �����","������� ���������� ���������� ��������","�������� ��� ���������� � ���������� ������"}),  
    @dog6  : dog(breed: "������������", sizemin: 66, sizemax: 76,weightmin: 50, weightmax:58, description: string{"��������� �������� �������","������ ��������� ��� ��������� � ������������ ������","������-������","�������� ������ �������� � ����� ������","������, �������� ������������� ����"}),  
    @dog7  : dog(breed: "���������", sizemin: 65, sizemax: 70,weightmin:50, weightmax:70, description: string{"��������� �������� �������","������ ��������� ��� ��������� � ������������ ������","������-������","�������� ������ �������� � ����� ������"}),  
    @dog8  : dog(breed: "���-���", sizemin: 48, sizemax: 51,weightmin:21, weightmax:32, description: string{"��������� �������� �������","�������� ��� ���������� � ���������� ������","��������� ����������� ���� �� �������","������� �� ������ ����","�������� ������� ������"}),  
    @dog9  : dog(breed: "������������ �����-��������", sizemin: 35, sizemax: 38,weightmin:11, weightmax:13, description: string{"��������� �������� �������","��������� ����������� ���� �� �������","������� �� ������ ����","����������� ������"}),  
    @dog10 : dog(breed: "����������� ������", sizemin: 25, sizemax: 38,weightmin: 5, weightmax:8, description: string{"��������� �������� �������","������ ������������ ��� ����� � ��������","����������� ������","��������� ����������� ���� �� �������"}),  
    @dog11 : dog(breed: "����������� ������", sizemin: 38, sizemax: 45,weightmin:21, weightmax:34, description: string{"��������� �������� �������","��������� ����������� ���� �� �������","������-������","������","������� ���������� ���������� ��������"}),  
    @dog12 : dog(breed: "������-�����", sizemin: 35, sizemax: 36,weightmin:18, weightmax:27, description: string{"��������� �������� �������","������� �� ������ ����","��������� ���� �� �������","������� ���������� ���������� ��������","������"}),  
    @dog13 : dog(breed: "����������� ������", sizemin: 25, sizemax: 32,weightmin:3, weightmax:4, description: string{"��������� �������� �������","��������� ���� �� �������","������-��������","������������ ������"}),  
    @dog14 : dog(breed: "�������", sizemin: 20, sizemax: 22,weightmin:2, weightmax:5, description: string{"������������ ������","������ ������������ ��� ����� � ��������","�������� ��� ���������� � ���������� ������","��������� ����������� ���� �� �������"}),  
    @dog15 : dog(breed: "���������� �������", sizemin: 63, sizemax: 71,weightmin:46, weightmax:66, description: string{"������� ���������� ���������� ��������","�������� ������ �������� � ����� ������","����� ���� ������ ���������� ������������","����������� ������","�������� ���������� ������"}),
    @dog16 : dog(breed: "������� ������ ������", sizemin: 71, sizemax: 79, weightmin: 34, weightmax: 48, description: string{"������","��������� �������� �������","������� �� �������� �����","�������� ������ �������� � ����� ������"}),
    @dog17 : dog(breed: "���������� ������", sizemin: 68, sizemax: 76, weightmin: 27, weightmax: 32, description: string{"������","��������� �������� �������","������� ���������� ���������� ��������","������ ��� ������� ����������"}),
    @dog18 : dog(breed: "��������������� ������", sizemin: 56, sizemax: 71, weightmin: 25, weightmax: 30, description: string{"������","��������� �������� �������","������� �� �������� �����","�������� ������ �������� � ����� ������","������� �� ������ ����"}),
    @dog19 : dog(breed: "����������", sizemin: 53, sizemax: 56, weightmin: 24, weightmax: 28, description: string{"��������� ���� �� �������","������-��������","�������� ��� ���������� � ���������� ������"}),
    @dog20 : dog(breed: "�������", sizemin: 66, sizemax: 80, weightmin: 50, weightmax: 58, description: string{"��������� �������� �������","�������� ������ �������� � ����� ������","������-������","������ ��������� ��� ��������� � ������������ ������","������, �������� ������������� ����"}),
    @dog21 : dog(breed: "���������� ������", sizemin: 70, sizemax: 76, weightmin: 80, weightmax: 86, description: string{"��������� ���� �� �������","������� ���������� ���������� ��������","�������� ���������� ������","�������� ��� ���������� � ���������� ������"}),
    @dog22 : dog(breed: "���������� ����������", sizemin: 64, sizemax: 69, weightmin: 45, weightmax: 68, description: string{"��������� �������� �������","������� ���������� ���������� ��������","����� ���� ����������� ������","�������� ��� ���������� � ���������� ������"}),
    @dog23 : dog(breed: "���������", sizemin: 56, sizemax: 69, weightmin: 41, weightmax: 50, description: string{"������ ���� �������� �������� �����","��������� ���� �� �������","������� ���������� ���������� ��������","����� ���� ������ ���������� ������������","�������� ��� ���������� � ���������� ������"}),
    @dog24 : dog(breed: "���������� �����", sizemin: 46, sizemax: 56, weightmin: 23, weightmax: 30, description: string{"��������� �������� �������","�������� ������ �������� � ����� ������","�������� ��� ���������� � ���������� ������","��������� ����������� ���� �� �������","������ ������� ����"});

  $descMenu  := string{}; 
  $descChoise:= string{};

  // �������� ���� ��� ������ ��������� �����
  $Win_breed := MakeWindow("���������� ������ �����", 450, 20, 300, 430);
  TextColor($Win_breed, 3);
  $y_breed := 10;

  // �������� ���� ��� ������ ��������� �������������
  $Win_descs := MakeWindow("��������� �������������� �����", 20, 20, 420, 170);
  TextColor($Win_descs, 1);
  $y_desc := 10;

  // ��������� ������� �� ������ � ���� 
  if Ask("", "������ ����� �� ����� ������?") then
    $s_min := GetNumber(100, 100, "������ ������ � ����� (� ��)", "������� ����������� ������:", 1);
    $s_max := GetNumber(100, 100, "������ ������ � ����� (� ��)", "������� ������������ ������:", 100);
    OutText($Win_descs,10,$y_desc,"���� �� "+ToString($s_min)+" �� "+ToString($s_max)+" ��");
    $y_desc:=$y_desc+15;
  else
    $s_min := 1; 
    $s_max := 100;
  end;
  if Ask("", "������ ����� �� ���� ������?") then
    $w_min := GetNumber(100, 100, "��� ������ (� ��)", "������� ����������� ���:", 1);
    $w_max := GetNumber(100, 100, "��� ������ (� ��)", "������� ������������ ���:", 100);
    OutText($Win_descs,10,$y_desc,"��� �� "+ToString($w_min)+" �� "+ToString($w_max)+" ��");
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
    $n := Menu(20, 200, "�������� �������������� ������", $d_Menu, 0);
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
  Message("", "����� ��������");
  CloseWindow($Win_breed);
  CloseWindow($Win_descs);
end.