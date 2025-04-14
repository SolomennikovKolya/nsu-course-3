
phi = [linspace(0.01, %pi/6 - 0.01, 200), 
       linspace(%pi/3 + 0.01, %pi/2 - 0.01, 200)];
rho = 2*tan(3*phi);

// Фильтрация
mask = rho < 200;
phi = phi(mask);
rho = rho(mask);

/*
// Построение графика в полярных координатах
clf();
polarplot(phi, rho, 'r-');
a = gca();
a.grid = [1 1];
*/

// Преобразуем в декартовы координаты
x = rho .* cos(phi);
y = rho .* sin(phi);

// Построение графика
plot(x, y, 'r-', 'LineWidth', 1);
a = gca();
a.isoview = "on"; // Сохраняет пропорции осей

