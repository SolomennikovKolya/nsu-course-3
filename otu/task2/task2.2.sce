
/*
phi_segments = [
    linspace(0/6*%pi, 1/6*%pi, 100),
    linspace(2/6*%pi, 3/6*%pi, 100),
    linspace(4/6*%pi, 5/6*%pi, 100),
    linspace(6/6*%pi, 7/6*%pi, 100),
    linspace(8/6*%pi, 9/6*%pi, 100),
    linspace(10/6*%pi, 11/6*%pi, 100)];

rho_segments = list();
for i = 1:6
    rho_segments(i) = 2 * tan(3 * phi_segments(i));
end

phi_combined = [];
rho_combined = [];
for i = 1:6
    phi_combined = [phi_combined, phi_segments(i), %nan];
    rho_combined = [rho_combined, rho_segments(i), %nan];
end

polarplot(phi_combined, rho_combined);
*/


/*phi_combined = [];
rho_combined = [];
for i = 1:6
    phi_combined = [phi_combined, phi_segments(i), %nan];
    rho_combined = [rho_combined, rho_segments(i), %nan];
end

polarplot(phi_combined, rho_combined);
*/

phi = linspace(-%pi, %pi, 1000);
phi = phi(2*tan(3*phi) > 0);
phi = phi(2*tan(3*phi) < 20);

rho = 2*tan(3*phi);
polarplot(phi, rho);

/*
clf(); // Очистка графика

// Определяем диапазон углов, исключая асимптоты
phi = linspace(-%pi, %pi, 5000); // Больше точек для гладкости
phi = phi(2*tan(3*phi) > 0);

// Находим точки разрыва (где cos(3*phi) = 0 → tan(3*phi) = ±inf)
asymptotes = ( (3*phi + %pi/2) ./ %pi ) == round( (3*phi + %pi/2) ./ %pi );
phi_clean = phi(~asymptotes); // Удаляем точки разрывов

// Разбиваем на непрерывные участки между разрывами
breaks = find(abs(diff(tan(3*phi_clean))) > 10); // Резкие скачки = разрывы
phi_segments = [];
rho_segments = [];

start_idx = 1;
for i = 1:length(breaks)
    end_idx = breaks(i);
    segment_phi = phi_clean(start_idx:end_idx);
    segment_rho = 2 * tan(3 * segment_phi);
    phi_segments = [phi_segments, segment_phi, %nan]; // %nan разрывает линии
    rho_segments = [rho_segments, segment_rho, %nan];
    start_idx = end_idx + 1;
end

// Добавляем последний сегмент
segment_phi = phi_clean(start_idx:$);
segment_rho = 2 * tan(3 * segment_phi);
phi_segments = [phi_segments, segment_phi];
rho_segments = [rho_segments, segment_rho];

polarplot(phi_segments, rho_segments);
*/

/*
phi = [linspace(0.01, %pi/6 - 0.01, 200), 
       linspace(%pi/3 + 0.01, %pi/2 - 0.01, 200)];
rho = 2*tan(3*phi);

// Фильтрация
mask = rho < 200;
phi = phi(mask);
rho = rho(mask);

clf();
polarplot(phi, rho);
a = gca();
a.grid = [1 1];
*/

/*
// Преобразуем в декартовы координаты
x = rho .* cos(phi);
y = rho .* sin(phi);

// Построение графика
plot(x, y, 'r-', 'LineWidth', 1);
a = gca();
a.isoview = "on"; // Сохраняет пропорции осей
*/
