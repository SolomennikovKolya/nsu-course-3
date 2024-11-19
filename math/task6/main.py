from scipy.integrate import quad
import math
import sys

# Формула трапеций
def trapezoidal_integral(A, B, func, N):
    result = 0
    step = (B - A) / N
    for i in range(N):
        xi = A + i * step
        xi_1 = A + (i + 1) * step
        result += (xi_1 - xi) / 2 * (func(xi) + func(xi_1))
    return result

# Формула парабол
def simpson_integral(A, B, func, N):
    result = 0
    step = (B - A) / N
    for i in range(N):
        xi = A + i * step
        xi_1 = A + (i + 1) * step
        result += (xi_1 - xi) / 6 * (func(xi) + 4 * func((xi + xi_1) / 2) + func(xi_1))
    return result

# Формула Симпсона 3/8
def simpson38_integral(A, B, func, N):
    result = 0
    step = (B - A) / N
    for i in range(N):
        xi = A + i * step
        xi_1 = A + (i + 1) * step
        result += (xi_1 - xi) / 8 * (func(xi) + 3 * func((2 * xi + xi_1) / 3) + 3 * func((xi + 2 * xi_1) / 3) + func(xi_1))
    return result

# Порядок точности по правилу Рунге
def runge(s1, s2, s3):
    return math.log2(abs((s1 - s2) / (s2 - s3)))

# Оценка ошибки методом Рунге (p - порядок точности)
def runge_error(s1, s2, p):
    return abs(s1 - s2) / (2 ** p - 1)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <number_of_segments>")
        return 1
    
    A, B = 5.0, 7.0                            # Границы интегрирования
    N = int(sys.argv[1])                       # Количество элементарных отрезков на [a, b]
    func = lambda x: math.exp(x) * math.cos(x) # Подынтегральная фуникция
    
    # Реальное значение интеграла
    real_int, real_int_error = quad(func, A, B)
    print(f"Реальное значение интеграла: {real_int:.15f} ± {real_int_error:.15f}")
    print("")
    
    # Формула трапеций
    trapezoidal_s1 = trapezoidal_integral(A, B, func, N)
    trapezoidal_s2 = trapezoidal_integral(A, B, func, N * 2)
    trapezoidal_s3 = trapezoidal_integral(A, B, func, N * 4)
    trapezoidal_runge = runge(trapezoidal_s1, trapezoidal_s2, trapezoidal_s3)
    trapezoidal_runge_error = runge_error(trapezoidal_s1, trapezoidal_s2, trapezoidal_runge)
    print(f"Формула трапеций:       {trapezoidal_s1:.15f}")
    print(f"Разность:               {abs(real_int - trapezoidal_s1):.15f}")
    print(f"Порядок точности:       {trapezoidal_runge:.15f}")
    print(f"Оценка ошибки:          {trapezoidal_runge_error:.15f}")
    print("")

    # Формула Симпсона
    simpson_s1 = simpson_integral(A, B, func, N)
    simpson_s3 = simpson_integral(A, B, func, N * 2)
    simpson_s2 = simpson_integral(A, B, func, N * 4)
    simpson_runge = runge(simpson_s1, simpson_s2, simpson_s3)
    simpson_runge_error = runge_error(simpson_s1, simpson_s2, simpson_runge)
    print(f"Формула Симпсона:       {simpson_s1:.15f}")
    print(f"Разность:               {abs(real_int - simpson_s1):.15f}")
    print(f"Порядок точности:       {simpson_runge:.15f}")
    print(f"Оценка ошибки:          {simpson_runge_error:.15f}")
    print("")

    # Формула Симпсона 3/8
    simpson38_s1 = simpson38_integral(A, B, func, N)
    simpson38_s2 = simpson38_integral(A, B, func, N * 2)
    simpson38_s3 = simpson38_integral(A, B, func, N * 4)
    simpson38_runge = runge(simpson38_s1, simpson38_s2, simpson38_s3)
    simpson38_runge_error = runge_error(simpson38_s1, simpson38_s2, simpson38_runge)
    print(f"Формула Симпсона 3/8:   {simpson38_s1:.15f}")
    print(f"Разность:               {abs(real_int - simpson38_s1):.15f}")
    print(f"Порядок точности:       {simpson38_runge:.15f}")
    print(f"Оценка ошибки:          {simpson38_runge_error:.15f}")
    print("")

if __name__ == "__main__":
    main()
