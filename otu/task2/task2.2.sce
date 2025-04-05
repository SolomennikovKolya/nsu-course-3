
phi = linspace(0, 2*%pi, 1000); // 1000 точек от 0 до 2π
rho = 2*tan(3*phi); // Вычисляем значения rho

// Удаляем бесконечные значения около асимптот
// threshold = 10;
// rho(abs(rho) > threshold) = %nan;

// Построение графика в полярных координатах
/*
clf(); // Очистка предыдущих графиков
polarplot(phi, rho, 'r-', 'LineWidth', 2);
title('Полярный график $\rho(\phi) = 2\tan(3\phi)$', 'fontsize', 4);
legend('$\rho = 2\tan(3\phi)$', 2);
a = gca();
a.grid = [1 1]; // Включаем сетку
*/

// Преобразуем в декартовы координаты
x = rho .* cos(phi);
y = rho .* sin(phi);

// Построение графика
plot(x, y, 'r-', 'LineWidth', 1);
title('$\rho(\phi) = 2\tan(3\phi)$', 'fontsize', 4);
xlabel('x');
ylabel('y');
a = gca();
a.isoview = "on"; // Сохраняет пропорции осей

// Добавляем полярную сетку вручную
theta = linspace(0, 2*%pi, 36);
for r = 1:5
    xp = r * cos(theta);
    yp = r * sin(theta);
    plot(xp, yp, ':', 'color', color('gray'));
end
