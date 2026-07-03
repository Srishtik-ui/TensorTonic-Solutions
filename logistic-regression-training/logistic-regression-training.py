import numpy as np

def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_regression(X, y, lr, steps):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        
        z = X @ w + b
        p = _sigmoid(z)

        error = p - y
        dw = (X.T @ error) / N
        db = np.mean(error)
        w -= lr * dw
        b -= lr * db

    return w, b