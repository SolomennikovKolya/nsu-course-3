
u = linspace(0, 4*%pi, 50);  
v = linspace(0, 6*%pi, 50);  

// Создание сетки
[U, V] = ndgrid(u, v);

// Вычисление координат
X = cos(U) .* U .* (1 + cos(V) / 2);
Y = (U / 2) .* sin(V);
Z = sin(U) .* U .* (1 + cos(V) / 2);

// Построение 3D графика
plot3d2(X, Y, Z);
xlabel("X");
ylabel("Y");
zlabel("Z");
title("Параметрическая поверхность");
