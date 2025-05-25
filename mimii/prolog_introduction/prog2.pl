word(article, a).
word(article, every).
word(noun, criminal).
word(noun, 'big kahuna burger').
word(verb, eats).
word(verb, likes).

sentence(W1, W2, W3, W4, W5) :-
    word(article, W1),
    word(noun, W2),
    word(verb, W3),
    word(article, W4),
    word(noun, W5).

% Предикат для форматирования предложения
format_sentence(W1, W2, W3, W4, W5, Sentence) :-
    atomic_list_concat([W1, W2, W3, W4, W5], ' ', Sentence).

print_all_sentences :-
    sentence(W1, W2, W3, W4, W5),
    format_sentence(W1, W2, W3, W4, W5, Sentence),
    writeln(Sentence),
    fail.

% Запросы
% findall(Sentence, sentence(W1, W2, W3, W4, W5), Sentences).
% print_all_sentences.
