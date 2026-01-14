import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_test = np.asarray(x_test, dtype=float)

    if x_train.ndim != 2 or x_train.shape[1] != 2:
        raise ValueError("x_train must have shape (n_samples, 2)")
    if x_test.ndim != 2 or x_test.shape[1] != 2:
        raise ValueError("x_test must have shape (n_samples, 2)")
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("y_train length must match x_train samples")

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    X = (x_train - mu) / sigma
    Xt = (x_test - mu) / sigma

    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    Xtb = np.c_[np.ones((Xt.shape[0], 1)), Xt]

    w = np.zeros(Xb.shape[1], dtype=float)

    lr = 0.1
    iters = 2000

    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        grad = (Xb.T @ (p - y_train)) / Xb.shape[0]
        w -= lr * grad

    zt = Xtb @ w
    pt = 1.0 / (1.0 + np.exp(-np.clip(zt, -50, 50)))
    return (pt >= 0.5).astype(int)
    pass

