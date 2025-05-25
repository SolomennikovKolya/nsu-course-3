word(abalone,a,b,a,l,o,n,e).
word(abandon,a,b,a,n,d,o,n).
word(enhance,e,n,h,a,n,c,e).
word(anagram,a,n,a,g,r,a,m).
word(connect,c,o,n,n,e,c,t).
word(elegant,e,l,e,g,a,n,t).

crosswd(V1, V2, V3, H1, H2, H3) :-
    word(V1, _, V1_2, _, V1_4, _, V1_6, _),
    word(V2, _, V2_2, _, V2_4, _, V2_6, _),
    word(V3, _, V3_2, _, V3_4, _, V3_6, _),
    word(H1, _, H1_2, _, H1_4, _, H1_6, _),
    word(H2, _, H2_2, _, H2_4, _, H2_6, _),
    word(H3, _, H3_2, _, H3_4, _, H3_6, _),
    
    % Условия пересечения
    V1_2 = H1_2, V1_4 = H2_2, V1_6 = H3_2,
    V2_2 = H1_4, V2_4 = H2_4, V2_6 = H3_4,
    V3_2 = H1_6, V3_4 = H2_6, V3_6 = H3_6.

% Запрос