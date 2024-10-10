from typing import List, Tuple

N: int = 5
EPSILON: float = 0.1
GAMMA: float = 0.1

def getTestData(testNum: int) -> Tuple[List[List[float]], List[float]]:
    A = [[0.0] * N for i in range(N)]
    f = [0.0] * N
    
    match testNum:
        case 1:
            for row in range(N):
                for col in range(N):
                    if row == col:
                        A[row][col] = 2.0
                    elif row == col + 1 or row + 1 == col:
                        A[row][col] = -1.0
            for i in range(N):
                f[i] = 2.0
        
        case 2:
            for row in range(N):
                for col in range(N):
                    if row == col:
                        A[row][col] = 2.0
                    elif row == col + 1 or row + 1 == col:
                        A[row][col] = -1.0
            for i in range(N):
                f[i] = 2.0 + EPSILON
        
        case 3:
            for row in range(N):
                for col in range(N):
                    if row == col:
                        A[row][col] = float(2 * (row + 1)) + GAMMA
                    elif row == col + 1 or row + 1 == col:
                        A[row][col] = -1.0
            for i in range(N):
                f[i] = 2.0 * (row + 2.0) + GAMMA

    return A, f

def printMatrix(A: List[List[float]]) -> None:
    s = [[str(e) for e in row] for row in A]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

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
    testNum = int(input("Номер теста: "))
    A, f = getTestData(testNum)
    x = calculateX(A, f)
    print(x)
