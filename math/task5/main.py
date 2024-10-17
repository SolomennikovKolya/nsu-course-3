import numpy as np
from scipy import linalg

N: int
EPSILON: float

def mod(x):
    return np.sum(np.square(x))

def iterativeMethod(A, f, x0, B, tau):
    xk = x0
    invB = np.linalg.inv(B)
    
    x = linalg.solve(A, f)
    zk = xk - x
    rk = A * xk - f
    k = 0
    
    p = 3
    sep = "\t"
    print("k", "xk[0]", "xk[1]", "mod(zk)", "< eps", "mod(rk)", "< eps", sep=sep)
    
    while (mod(rk) >= EPSILON or mod(rk) >= EPSILON) and k < 100:
        zk = xk - x
        rk = A @ xk - f
        print(k, round(xk[0], p), round(xk[1], p), round(mod(zk), p), mod(zk) < EPSILON, round(mod(rk), p), mod(rk) < EPSILON, sep=sep)
        
        xk = xk - tau * invB @ (A @ xk - f)
        k += 1
    
    return x

def JacobiMethod(A, f, x0):
    B = np.diag(A.diagonal())
    tau = 1
    return iterativeMethod(A, f, x0, B, tau)

def SeidelMethod(A, f, x0):
    B = np.tril(A)
    tau = 1
    return iterativeMethod(A, f, x0, B, tau)

if __name__ == "__main__":
	A = np.array([[3, 2], [1, 3]])
	f = np.array([-1, 2])
	x0 = np.array([0, 0])
	EPSILON = 0.0001
 
	print("Jacobi Method:")
	x = JacobiMethod(A, f, x0)
	print("Seidel Method:")
	x = SeidelMethod(A, f, x0)
	# print(x)
