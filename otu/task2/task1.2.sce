
A = [1, 4, 2; 2, 1, -2; 0, 1, -1];
B = [4, 6, -2; 4, 10, 1; 2, 4, -5];

// D = 2A - (A^2 + B)B
D = 2*A - (A*A + B) * B;
disp("Матрица D:");
disp(D);

// Вычисляем обратную матрицу
det_D = det(D);
if abs(det_D) < 1e-10 then
    disp("Обратной матрицы не существует (D вырождена)");
else
    D_inv = inv(D);
    disp("D^{-1}:");
    disp(D_inv);
    
    // Проверка: D * D^{-1} = I
    I = D * D_inv;
    disp("I = D*D^{-1}:");
    disp(I);
end
