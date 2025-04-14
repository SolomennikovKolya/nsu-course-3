
t = linspace(0, 2*%pi, 100);

// Первая кривая
x1 = sin(t);
y1 = sin(2*t);
z1 = t / 5;

// Вторая кривая
x2 = cos(t);
y2 = cos(2*t);
z2 = sin(t);

// Построение линий
clf();
param3d(x1, y1, z1);
param3d(x2, y2, z2);
title("Параметрические кривые");
