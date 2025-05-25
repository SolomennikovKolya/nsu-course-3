:- encoding(utf8).

/* Вводим предикаты базы данных */
:- dynamic db_yes/1, db_no/1.

/* Вводим правила высокого уровня */
rule(1, "пингвин", [2, 5, 6]).
rule(2, "медведь", [1, 7, 8]).

/* Признаки животных */
property(1, "имеет шерсть").
property(2, "имеет перья").
property(3, "имеет плавники").
property(4, "летает").
property(5, "плавает").
property(6, "имеет черно-белый цвет").
property(7, "имеет бурый цвет").
property(8, "лазает по деревьям").

/* Основные правила механизма вывода */
animal(X) :- rule(_, X, Property), check_property(Property).
animal(_) :- write("Такого животного я не знаю."), fail.
check_property([N | Property]) :- property(N, A), yes(A), check_property(Property).
check_property([]).

/* Правила проверки признаков */
yes(X) :- db_yes(X), !.
yes(X) :- not(no(X)), !, check_if(X).
no(X) :- db_no(X), !.

/* Опрос пользователя */
check_if(X) :- write("Оно "), write(X), writeln(" ?"), read(Reply), remember(Reply, X).
remember(yes, X) :- asserta(db_yes(X)).
remember(no, X) :- asserta(db_no(X)), fail.

/* Целевое правило */
game :- retractall(db_yes(_)), retractall(db_no(_)), animal(X), write("Задуманное Вами животное - "), write(X).
