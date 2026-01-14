import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    learning_rate = 0.01
    iterations = 10000

    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    y_train_r = y_train.reshape(-1, 1)
    x_train_bias = np.hstack([np.ones((m_train, 1)), x_train])
    x_test_bias = np.hstack([np.ones((m_test, 1)), x_test])
    weights = np.zeros((x_train_bias.shape[1], 1))

    for _ in range(iterations):
        linear_model = np.dot(x_train_bias, weights)
        y_predicted = 1 / (1 + np.exp(-linear_model))

        gradient = np.dot(x_train_bias.T, (y_predicted - y_train_r)) / m_train
        weights -= learning_rate * gradient

    linear_pred = np.dot(x_test_bias, weights)
    y_prob = 1 / (1 + np.exp(-linear_pred))

    y_pred = (y_prob >= 0.5).astype(int).flatten()

    return y_pred
    pass
