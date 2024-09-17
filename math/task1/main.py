import math

a: float = -1
b: float = -8
c: float = 6
e: float = 0.0001

delta: float = 1000

# Подставляет значение x в полином x^3 + ax^2 + bx + c
def f(x: float) -> float:
    return x ** 3 + a * x ** 2 + b * x + c

# Находит корень на отрезке [l, r] с точностью e, если известно, что на этом отрезке ровно 1 корень
def bisection(l: float, r: float) -> float:
    if (l > r):
        return bisection(r, l)
    
    if l == -math.inf and r == math.inf:
        if f(0) > e:
            return bisection(l, 0)
        elif f(0) < -e:
            return bisection(0, r)
        else:
            return 0
            
    if l == -math.inf:
        if f(r - delta) < -e:
            return bisection(r - delta, r)
        else:
            return bisection(-math.inf, r - delta)
    
    if r == math.inf:
        if f(l + delta) > e:
            return bisection(l, l + delta)
        else:
            return bisection(l + delta, math.inf)
    
    m = (l + r) / 2
    if f(l) < f(r):
        if f(m) < -e:
            return bisection(m, r)
        elif f(m) > e:
            return bisection(l, m)
        else:
            return m
    else:
        if f(m) < -e:
            return bisection(l, m)
        elif f(m) > e:
            return bisection(m, r)
        else:
            return m

# Находит все корни полинома
def calculate() -> None:
    # Дискриминант уравнения f'(x) = 3x^2 + 2ax + b = 0
    d: float = 4 * a ** 2 - 12 * b
    
    if d < -e:
        if f(0) < -e:
            x1 = bisection(0, math.inf)
            print("x1 = " + str(x1) + " кратности 1")
        elif f(0) > e:
            x1 = bisection(-math.inf, 0)
            print("x1 = " + str(x1) + " кратности 1")
        else:
            x1 = 0
            print("x1 = " + str(x1) + " кратности 1")
    
    elif -e <= d <= e:
        t = -a / 3
        if f(t) < -e:
            x1 = bisection(t, math.inf)
            print("x1 = " + str(x1) + " кратности 1")
        elif f(t) > e:
            x1 = bisection(-math.inf, t)
            print("x1 = " + str(x1) + " кратности 1")
        else:
            x1 = t
            print("x1 = " + str(x1) + " кратности 3")
    
    else:
        q1 = (-a - math.sqrt(a * a - 3 * b)) / 3
        q2 = (-a + math.sqrt(a * a - 3 * b)) / 3
        
        if f(q1) > e and f(q2) < -e:
            x1 = bisection(-math.inf, q1)
            x2 = bisection(q1, q2)
            x3 = bisection(q2, math.inf)
            print("x1 = " + str(x1) + " кратности 1")
            print("x2 = " + str(x2) + " кратности 1")
            print("x3 = " + str(x3) + " кратности 1")
            
        elif f(q1) > e and -e <= f(q2) <= e:
            x1 = q2
            x2 = bisection(-math.inf, q1)
            print("x1 = " + str(x1) + " кратности 2")
            print("x2 = " + str(x2) + " кратности 1")
        
        elif -e <= f(q1) <= e and f(q2) < -e:
            x1 = q1
            x2 = bisection(q2, math.inf)
            print("x1 = " + str(x1) + " кратности 2")
            print("x2 = " + str(x2) + " кратности 1")
        
        elif f(q1) > e and f(q2) > e:
            x1 = bisection(-math.inf, q1)
            print("x1 = " + str(x1) + " кратности 1")
        
        elif f(q1) < -e and f(q2) < -e:
            x1 = bisection(q2, math.inf)
            print("x1 = " + str(x1) + " кратности 1")
        
        elif -e <= f(q1) <= e and -e <= f(q2) <= e:
            x1 = (q1 + q2) / 2
            print("x1 = " + str(x1) + " кратности 3")


if __name__ == "__main__":
    test_data = [
        [-2, -5, 6],    # Три разных корня x1 = -2, x2 = 1, x3 = 3
        [-7, 16, -12],  # x1 = 2 кратности 2, x2 = 3 кратности 1
        [-3, 3, -1],    # x1 = 1 кратности 3
        [-5, 9, -5],    # x1 = 1 кратности 1 (x2 = 2 + i, x3 = 2 - i)
    ]
    
    for i in range(len(test_data)):
        a, b, c = test_data[i][0], test_data[i][1], test_data[i][2]
        print("iter " + str(i) + ":")
        calculate()
        print()
