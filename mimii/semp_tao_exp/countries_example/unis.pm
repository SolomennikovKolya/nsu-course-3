module Unis;

var $c_min, $c_max, $b_min, $b_max, $y_name, $y_desc, $countries : integer;
var $Win_name, $Win_descs : integer;
var $descMenu, $descChoise : set of string;
var $d_Menu : tuple of string;
var $ch_desc : string;
var $min_budget_places, $max_budget_places, $min_temp, $max_temp : integer;
var $selected_name : string;

//--------------------------------------------------------------------------------------------------
function SetToTuple( $S: set of string ): tuple of string
begin
var $t: tuple of string := string[];
  for $e in $S loop
   $t:=$t+string[$e];
  end;
  return $t;
end;

//--------------------------------------------------------------------------------------------------
rule Stop
=>
  activate group();
end;

rule Print_Country
   forall $d: country(name: $name, temp: $temp, budget: $budget_places, description: $desc)  when
   ($name = $selected_name)
=>
  OutText( $Win_name, 10, $y_name, $name);
  $y_name := $y_name + 30;

  OutText( $Win_name, 10, $y_name, "Budget: " + ToString($budget_places));
  $y_name := $y_name + 30;

  OutText( $Win_name, 10, $y_name, "Temp: " + ToString($temp));
  $y_name := $y_name + 30;

  OutText( $Win_name, 10, $y_name, "Features:");
  $y_name := $y_name + 15;

  for $d in $desc loop
  	OutText( $Win_name, 10, $y_name, ToString($d));
  	$y_name := $y_name + 15;
  end;

  $y_name := $y_name + 15;
end;

rule Print_Countries
   forall country(name: $nam)
=>
  $descMenu:=$descMenu+string{$nam}-$descChoise;
end;

//-------------------------------Display list of selected countries--------------------------------
rule Print_Names
   forall $d: country(name: $name, temp: $temp, budget: $budget_places, description: $desc)  when
   ($temp <= $c_max) & ($temp >= $c_min) & ($budget_places <= $b_max) & ($budget_places >= $b_min) & ($descChoise <= $desc)
=>
var $s : set of string;
var $flag : integer;

  $countries:=$countries +1;
  OutText( $Win_name, 10, $y_name, ToString($name) + "; Features:");
  $y_name := $y_name + 15;

  for $d in $desc loop
  	OutText( $Win_name, 10, $y_name, ToString($d));
  	$y_name := $y_name + 15;
  end;

  $y_name := $y_name + 15;

  $descMenu:=$descMenu+$desc-$descChoise;
end;

//------------------------------Update descMenu set and display selection-------------------------------------
rule New_Descs
=>
  $descChoise:=$descChoise+string{$ch_desc};
  OutText($Win_descs,10,$y_desc,$ch_desc);
  $y_desc:=$y_desc+15;
end;

//--------------------------------------------------------------------------------------------------
begin
var $n : integer;

         new
            @country1   : country(name: "Russia", temp: 2, budget: 650, description: string{"Moscow", "safely", "museums", "mountains"}),
            @country2   : country(name: "France", temp: 5, budget: 1200, description: string{"Paris", "safely", "museums", "meal", "mountains"}),
            @country3   : country(name: "England", temp: 4, budget: 1420, description: string{"London", "safely", "museums"}),
            @country4   : country(name: "Egypt", temp: 8, budget: 410, description: string{"Cairo", "beach", "museums", "deserts"}),
            @country5   : country(name: "UAE", temp: 9, budget: 1130, description: string{"Dubai", "safely", "beach", "deserts"}),
            @country6   : country(name: "Pakistan", temp: 9, budget: 220, description: string{"Islamabad", "deserts"}),
            @country7   : country(name: "Thailand", temp: 9, budget: 430, description: string{"Bangkok", "safely", "beach", "museums"}),
            @country8   : country(name: "Japan", temp: 7, budget: 1050, description: string{"Tokyo", "safely", "beach", "museums", "meal"}),
            @country9   : country(name: "USA", temp: 5, budget: 1650, description: string{"Washington", "safely", "beach", "museums", "deserts", "mountains"}),
            @country10  : country(name: "Brazil", temp: 8, budget: 650, description: string{"Brazilia", "safely", "beach"}),
            @country11  : country(name: "Canada", temp: 2, budget: 1050, description: string{"Toronto", "safely"}),
            @country12  : country(name: "Mexico", temp: 7, budget: 560, description: string{"Mexico", "beach", "museums", "deserts"}),
            @country13  : country(name: "Iceland", temp: 4, budget: 1350, description: string{"Reykjavík", "safely"}),
            @country14  : country(name: "Ireland", temp: 4, budget: 1050, description: string{"Dublin", "safely"}),
            @country15  : country(name: "Cuba", temp: 7, budget: 410, description: string{"Havana", "safely", "beach"});

  $descMenu  := string{};
  $descChoise:= string{};

  $Win_name := MakeWindow( "Matching countries", 450, 20, 300, 430 );
  TextColor($Win_name, 3);
  $y_name := 10;

  $Win_descs := MakeWindow( "Selected features", 20, 20, 420, 170 );
  TextColor($Win_descs, 1);
  $y_desc := 10;

  $min_budget_places:=1;
  $max_budget_places:=10000;
  $min_temp:=1;
  $max_temp:=10;

  if Ask( "", "Need information about a country without filtering parameters?" ) then
    call group(Print_Countries,Stop);

    $d_Menu:=SetToTuple($descMenu);
    $n:=1;
    $n := Menu( 20, 200, "Select country", $d_Menu, 0 );
    if $n!=0 then
      WriteLn($d_Menu);
      WriteLn($d_Menu[$n]);
      $selected_name := $d_Menu[$n];
      call group(Print_Country,Stop);
    end;
    Message( "", "Selection completed" );
  else
    if Ask( "", "Filter by number of temp?" ) then
      $c_min := GetNumber( 100, 100, "Number of temp", "Enter minimum value from " + ToString($min_temp) + ":", $min_temp );
      $c_max := GetNumber( 100, 100, "Number of temp", "Enter maximum value up to " + ToString($max_temp) + ":", $max_temp );

      if $c_min < $min_temp then
        $c_min := $min_temp;
      end;
      if $c_max > $max_temp then
	  $c_max := $max_temp;
      end;

      OutText($Win_descs,10,$y_desc,"Number of temp from "+ToString($c_min)+" to "+ToString($c_max));
      $y_desc:=$y_desc+15;
    else
      $c_min := $min_temp;
      $c_max := $max_temp;
    end;

    if Ask( "", "Filter by number of budget?" ) then
      $b_min := GetNumber( 100, 100, "Number of budget", "Enter minimum value from " + ToString($min_budget_places) + ":", $min_budget_places );
      $b_max := GetNumber( 100, 100, "Number of budget", "Enter maximum value up to " + ToString($max_budget_places) + ":", $max_budget_places );

      if $b_min < $min_budget_places then
        $b_min := $min_budget_places;
      end;
      if $b_max > $max_budget_places then
        $b_max := $max_budget_places;
      end;

      OutText($Win_descs,10,$y_desc,"Number of budget from "+ToString($b_min)+" to "+ToString($b_max));
      $y_desc:=$y_desc+15;
    else
      $b_min := $min_budget_places;
      $b_max := $max_budget_places;
    end;

    $countries:=0;
    call group(Print_Names,Stop);
    $n:=1;
    while ($n!=0 & $countries>1) loop
      $d_Menu:=SetToTuple($descMenu);
      $n := Menu( 20, 200, "Select country features", $d_Menu, 0 );
      if $n!=0 then
        WriteLn($d_Menu);
        $ch_desc:=$d_Menu[$n];
        WriteLn($ch_desc);
        ClearWindow($Win_name);
        $y_name:=10;
        $descMenu:=string{};
        $countries:=0;
        call group(New_Descs,Print_Names,Stop);
        if #$descMenu=0 then $n:=0; end;
      end;
    end;
    Message( "", "Selection completed" );
  end;

  CloseWindow($Win_name);
  CloseWindow($Win_descs);
end.