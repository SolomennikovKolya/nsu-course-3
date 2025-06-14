n = 4;
T0 = 1.26;
T1 = 0;
K = 0.66;
Ti = 5.6;
h = 0.01;

// Определяем передаточную функцию
S = poly(0, 's');
W1 = (1 + 1/(Ti*S)) * K * 2/(1+T0*S)^n;
W=W1/(1+W1);

// Создаёт объект линейной системы непрерывного времени (tlist)
sl = syslin('c', W);
num = sl.num;
den = sl.den;
b = coeff(num)';
a = coeff(den)';

// Приводим систему в каноническую нормальную форму (форму Фробениуса)
order = length(a) - 1;
A = [zeros(1, order - 1); eye(order - 1, order - 1)];
A = [A, -a(1:$-1)];
B = [b; zeros(order - length(b) + 1, 1)];
C = [zeros(1, order), 1];
// disp(A);
// disp(B);
// disp(C);

// Дискретизирует непрерывную систему sl с шагом h с помощью Zero-Order Hold
// v_{k+1} = A_d * v_k + B_d * u_k; x_k = C_d * v_k + D_d * u_k
// Дискретизованные матрицы, хранящиеся в структуре dMat
dMat = dscr(sl, h);
t = [0:h:100-h];
v = zeros(dMat.B); // Нулнвое начальное состояние (система в покое)
u = ones(t);       // Входной сигнал - единичный скачок (u_k = 1 для всех k)
x = zeros(t);      // Инициализация массива для выходного сигнала

// График непрерывной переходной характеристики
data = fscanfMat("./pi-regulator.TNO");
data_time = data(:, 1); // Временная шкала непрерывного отклика
data_y = data(:, 2);    // Значения непрерывного отклика
plot(data_time, data_y, 'red');

// График дискретной переходной характеристики
for i=1:length(u)
    v = dMat.A * v + dMat.B * u(i);
    x(i) = dMat.C * v + dMat.D * u(i);
end
plot(t, x,'blue');

// Вычисление ошибки
len = length(t);
err=0;
for l = 1:len
    [min_val, idx] = min(abs(data_time - t(l)));
    err = err + (x(l) - data_y(idx))^2;
end
err = sqrt(err/len);
disp(err);
