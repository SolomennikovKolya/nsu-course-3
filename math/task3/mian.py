from typing import List, Tuple

N: int
EPSILON: float
GAMMA: float

def getTestData(ci, di, ei, fi) -> Tuple[List[List[float]], List[float]]:
    A = [[0.0] * N for i in range(N)]
    f = [0.0] * N
    
    if GAMMA == 0:
        for i in range(2, N + 1):
            A[i - 1][i - 2] = ci
        for i in range(1, N + 1):
            A[i - 1][i - 1] = di
        for i in range(1, N):
            A[i - 1][i] = ei
        for i in range(N):
            f[i] = fi + EPSILON
    
    else:
        for row in range(N):
            for col in range(N):
                if row == col:
                    A[row][col] = float(2 * (row + 1)) + GAMMA
                elif row == col + 1 or row + 1 == col:
                    A[row][col] = -1.0
        for i in range(N):
            f[i] = 2.0 * (i + 2.0) + GAMMA

    return A, f

def calculateX(A: List[List[float]], f: List[float]) -> List[float]:
    c = [None, None] + [A[i - 1][i - 2] for i in range(2, N + 1)]
    d = [None] + [A[i - 1][i - 1] for i in range(1, N + 1)] + [None]
    e = [None] + [A[i - 1][i] for i in range(1, N)] + [None]
    
    a = [0.0] * (N + 1)
    b = [0.0] * (N + 1)
    a[2] = -e[1]/d[1]
    b[2] = f[0]/d[1]
    for i in range(2, N):
        a[i + 1] = -e[i] / (c[i] * a[i] + d[i])
        b[i + 1] = (f[i - 1] - c[i] * b[i]) / (c[i] * a[i] + d[i])
    
    x = [0.0] * (N + 1)
    x[N] = (f[N - 1] - c[N] * b[N]) / (c[N] * a[N] + d[N])
    for i in range(N - 1, 0, -1):
        x[i] = a[i + 1] * x[i + 1] + b[i + 1]
    
    return x[1:]

if __name__ == "__main__":
    
    N, EPSILON, GAMMA = 7, 0, 3
    ai, bi, ci, fi = -1, 2, -1, 2
    A, f = getTestData(ai, bi, ci, fi)
    
    x = calculateX(A, f)
    
    # print(x)
    for xi in x:
        print("{:.8f}".format(xi), end=" ")
