definition Langs;

class lang
  name: string;
  features: set of string;
  difficulty: integer(1..10);
  popularity: integer(1..100);
end;

end.