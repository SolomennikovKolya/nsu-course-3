n = 4;
T0 = 1.26;
K = 0.66;
Ti = 5.6;

// Передаточная функция
s = poly(0, 's');
W1 = (1 + 1/(Ti*s)) * K * 2*(1/(1+T0*s)^n);
W = W1/(1+W1);
disp("W: ", W);

// Система в форме канонической нормальной форме
sl = syslin('c', W);
num = sl.num;
den = sl.den;
b = coeff(num)';
a = coeff(den)';

order = length(a) - 1;
A = [zeros(1, order - 1); eye(order - 1, order - 1)];
A = [A, -a(1:$-1)];
B = [b; zeros(order - length(b) + 1, 1)];
C = [zeros(1, order), 1];
disp('A:', A);

// Находим H из уравнения Ляпунова для непрерывной системы
I = eye(A);
H = lyap(A, -I, 'c');
disp('H:', H);

// Спектр матрицы H
disp('Собственные числа H:', spec(H));

// Количественный запас устойчивости (максимальное собственное число)
disp('kappa:', norm(H, 2)); 
