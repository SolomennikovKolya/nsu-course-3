
t = linspace(0, 2*%pi, 100);

// Первая кривая
x1 = sin(t);
y1 = sin(2*t);
z1 = t / 5;

// Вторая кривая
x2 = cos(t);
y2 = cos(2*t);
z2 = sin(t);

// Построение 3D линий
clf();
param3d([x1; x2], [y1; y2], [z1; z2]);
xlabel("X");
ylabel("Y");
zlabel("Z");
title("Параметрические кривые");
