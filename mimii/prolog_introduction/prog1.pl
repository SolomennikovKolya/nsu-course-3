% Животные
animal(bootsy).
animal(corny).
animal(mack).
animal(flash).
animal(rover).
animal(spot).

% Типы животных (кошка или собака)
cat(bootsy).
cat(corny).
cat(mack).
dog(flash).
dog(rover).
dog(spot).

% Цвета животных 
color(bootsy, brown).
color(corny, black).
color(mack, ginger).
color(flash, spotted).
color(rover, ginger).
color(spot, white).

% Родословные
pedigree(X) :- (owns(tom, X); owns(kate, X)).

% Владельцы собак
owns(tom, X) :- animal(X), (color(X, black); color(X, brown)).
owns(kate, X) :- dog(X), color(X, Color), Color \= white, not(owns(tom, X)).
owns(alan, mack) :- not(owns(kate, bootsy)), not(pedigree(spot)).

% Животные без хозяина
animal_without_owner(X) :- animal(X), not(owns(_, X)).

% Зпросы
% animal_without_owner(X).
% owns(X, Y).
