
A = [-2, -1, 3, 2; -1, 1, 1, 0.6667; -3, -1, -1, 2; -3, -1, 3, -1];
b = [40; 20; 60; 60];

// Решаем систему Ax = b
x = A \ b;
disp("Решение системы:");
disp(x);

// Проверяем результат
b_calculated = A * x;
disp("Проверка:");
disp(b_calculated);
