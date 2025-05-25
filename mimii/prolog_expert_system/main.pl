:- encoding(utf8).
:- dynamic selected/2.
:- dynamic language/2.

% --- Языки программирования с оценками соответствия ---
% language(Name, [domain(DScore), platform(PScore), paradigm(ParScore)])
% Оценки — списки пар (Вариант, Балл от 0 до 10)

language(python, [
    domain([general_purpose-9, web-8, mobile-0, systems-0, data-10, games-4, academic-7]),
    platform([desktop-6, mobile-3, server-9, browser-0, native-0, cross_platform-7]),
    paradigm([procedural-7, oop-9, functional-3, logical-0])
]).

language(java, [
    domain([general_purpose-10, web-7, mobile-8, systems-0, data-0, games-0, academic-6]),
    platform([desktop-9, mobile-9, server-10, browser-0, native-0, cross_platform-8]),
    paradigm([procedural-0, oop-10, functional-2, logical-0])
]).

language(kotlin, [
    domain([general_purpose-8, web-3, mobile-9, systems-0, data-0, games-1, academic-4]),
    platform([desktop-6, mobile-10, server-4, browser-5, native-0, cross_platform-9]),
    paradigm([procedural-0, oop-9, functional-6, logical-0])
]).

language(c, [
    domain([general_purpose-0, web-0, mobile-0, systems-10, data-0, games-0, academic-3]),
    platform([desktop-0, mobile-0, server-0, browser-0, native-10, cross_platform-0]),
    paradigm([procedural-10, oop-0, functional-0, logical-0])
]).

language(cpp, [
    domain([general_purpose-9, web-2, mobile-5, systems-8, data-3, games-7, academic-6]),
    platform([desktop-9, mobile-6, server-8, browser-0, native-9, cross_platform-5]),
    paradigm([procedural-9, oop-8, functional-3, logical-0])
]).

language(csharp, [
    domain([general_purpose-9, web-6, mobile-7, systems-3, data-4, games-6, academic-5]),
    platform([desktop-8, mobile-7, server-7, browser-3, native-2, cross_platform-9]),
    paradigm([procedural-5, oop-10, functional-4, logical-0])
]).

language(rust, [
    domain([general_purpose-7, web-2, mobile-1, systems-9, data-0, games-3, academic-5]),
    platform([desktop-6, mobile-2, server-7, browser-0, native-10, cross_platform-4]),
    paradigm([procedural-7, oop-6, functional-5, logical-0])
]).

language(js, [
    domain([general_purpose-0, web-10, mobile-4, systems-0, data-2, games-5, academic-1]),
    platform([desktop-0, mobile-5, server-6, browser-10, native-0, cross_platform-9]),
    paradigm([procedural-2, oop-5, functional-3, logical-0])
]).

language(typescript, [
    domain([general_purpose-3, web-10, mobile-5, systems-0, data-2, games-3, academic-2]),
    platform([desktop-2, mobile-5, server-4, browser-10, native-0, cross_platform-9]),
    paradigm([procedural-3, oop-6, functional-5, logical-0])
]).

language(php, [
    domain([general_purpose-2, web-10, mobile-3, systems-0, data-2, games-1, academic-1]),
    platform([desktop-2, mobile-2, server-9, browser-8, native-0, cross_platform-6]),
    paradigm([procedural-4, oop-6, functional-2, logical-0])
]).

language(lua, [
    domain([general_purpose-2, web-0, mobile-2, systems-1, data-1, games-9, academic-2]),
    platform([desktop-4, mobile-3, server-3, browser-0, native-3, cross_platform-6]),
    paradigm([procedural-6, oop-4, functional-3, logical-0])
]).

language(r, [
    domain([general_purpose-3, web-0, mobile-0, systems-0, data-10, games-0, academic-8]),
    platform([desktop-8, mobile-0, server-6, browser-0, native-0, cross_platform-4]),
    paradigm([procedural-3, oop-2, functional-5, logical-0])
]).

language(julia, [
    domain([general_purpose-4, web-0, mobile-0, systems-2, data-9, games-1, academic-9]),
    platform([desktop-6, mobile-0, server-6, browser-0, native-0, cross_platform-5]),
    paradigm([procedural-3, oop-3, functional-7, logical-0])
]).

language(matlab, [
    domain([general_purpose-2, web-0, mobile-0, systems-1, data-9, games-0, academic-10]),
    platform([desktop-8, mobile-0, server-5, browser-0, native-0, cross_platform-3]),
    paradigm([procedural-3, oop-2, functional-6, logical-0])
]).

language(haskell, [
    domain([general_purpose-3, web-1, mobile-0, systems-1, data-4, games-0, academic-8]),
    platform([desktop-5, mobile-0, server-4, browser-0, native-0, cross_platform-3]),
    paradigm([procedural-0, oop-0, functional-10, logical-0])
]).

language(prolog, [
    domain([general_purpose-2, web-0, mobile-0, systems-0, data-1, games-0, academic-9]),
    platform([desktop-4, mobile-0, server-3, browser-0, native-0, cross_platform-2]),
    paradigm([procedural-0, oop-0, functional-0, logical-10])
]).

language(swift, [
    domain([general_purpose-7, web-1, mobile-10, systems-2, data-1, games-4, academic-3]),
    platform([desktop-6, mobile-10, server-3, browser-1, native-5, cross_platform-8]),
    paradigm([procedural-3, oop-8, functional-4, logical-0])
]).

language(elixir, [
    domain([general_purpose-2, web-7, mobile-1, systems-2, data-3, games-0, academic-6]),
    platform([desktop-3, mobile-2, server-7, browser-0, native-0, cross_platform-7]),
    paradigm([procedural-0, oop-1, functional-9, logical-0])
]).

language(erlang, [
    domain([general_purpose-2, web-6, mobile-1, systems-3, data-2, games-0, academic-6]),
    platform([desktop-3, mobile-2, server-8, browser-0, native-0, cross_platform-6]),
    paradigm([procedural-0, oop-1, functional-9, logical-0])
]).

% --- Критерии и опции ---
criteria([
    criterion(domain, 'Сфера использования языка:',
        [general_purpose, web, mobile, systems, data, games, academic]),
    criterion(platform, 'Целевая платформа:',
        [desktop, mobile, server, browser, native, cross_platform]),
    criterion(paradigm, 'Парадигма программирования:',
        [procedural, oop, functional, logical])
]).

% --- Запуск ---
start :-
    retractall(selected(_,_)),
    ask_all,
    rank_languages(Ranked),
    (   Ranked \= []
    ->  writeln('\nПодходящие языки:'),
        forall(member(Score-Lang, Ranked), format('~w (~d)~n', [Lang, Score]))
    ;   writeln('Не найдено подходящих языков.')
    ).

ask_all :-
    criteria(C),
    forall(member(criterion(Key, Question, Options), C), ask(Key, Question, Options)).

ask(Key, Question, Options) :-
    repeat,
        format('\n~w~n', [Question]),
        print_options(Options, 1),
        read(Index),
        (   integer(Index), Index > 0, length(Options, L), Index =< L
        ->  nth1(Index, Options, Value),
            asserta(selected(Key, Value)),
            !
        ;   writeln('Некорректный ввод, попробуйте снова.'), fail
        ).

print_options([], _).
print_options([H|T], N) :-
    format('~d. ~w~n', [N, H]),
    N1 is N + 1,
    print_options(T, N1).

rank_languages(RankedSorted) :-
    findall(Score-Name, (
        language(Name, [domain(Ds), platform(Ps), paradigm(Pars)]),
        selected(domain, DVal), memberchk(DVal-DScore, Ds),
        selected(platform, PVal), memberchk(PVal-PScore, Ps),
        selected(paradigm, ParVal), memberchk(ParVal-ParScore, Pars),
        DScore > 0, PScore > 0, ParScore > 0,
        Score is DScore + PScore + ParScore
    ), Ranked),
    sort(0, @>=, Ranked, RankedSorted).

% --- Добавление языка ---
lang_add :-
    write('Введите название нового языка: '), read(Name),
    criteria(C),
    collect_scores(C, [], FeatureLists),
    assertz(language(Name, FeatureLists)),
    writeln('Язык успешно добавлен.').

collect_scores([], Acc, Acc).
collect_scores([criterion(Key, Question, Options) | Rest], Acc, Result) :-
    format('\n~w~n', [Question]),
    collect_option_scores(Options, [], Pairs),
    Feature =.. [Key, Pairs],
    collect_scores(Rest, Acc, Result1), Result = [Feature | Result1].

collect_option_scores([], Acc, Acc).
collect_option_scores([Opt | T], Acc, Result) :-
    format('  Оцените степень соответствия "~w" (0-10): ', [Opt]),
    read(Score),
    integer(Score), Score >= 0, Score =< 10,
    collect_option_scores(T, [Opt-Score | Acc], Result).
collect_option_scores([Opt | T], Acc, Result) :-
    writeln('  Некорректная оценка, повторите ввод.'),
    collect_option_scores([Opt | T], Acc, Result).

% --- Справка (список всех языков) ---
lang_list :-
    findall(L, language(L, _), Langs),
    writeln('\nСписок всех языков:'),
    forall(member(L, Langs), (write('- '), writeln(L))).

% --- Информация о языке ---
lang_info(Name) :-
    language(Name, Properties),
    format('\nИнформация о языке ~w:~n', [Name]),
    criteria(CritList),
    forall(member(criterion(Key, Label, _), CritList), (
        member(Part, Properties),
        Part =.. [Key, Pairs],
        format('\n ~w~n', [Label]),
        print_score_table(Pairs)
    )).

print_score_table([]).
print_score_table([K-V | T]) :-
    format('  - ~w: ~d~n', [K, V]),
    print_score_table(T).
