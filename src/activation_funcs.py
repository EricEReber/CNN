def sigmoid(x):
    try:
        return 1.0 / (1 + np.exp(-x))
    except FloatingPointError:
        return np.where(x > np.zeros(x.shape), np.ones(x.shape), np.zeros(x.shape))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / (np.sum(np.exp(x), axis=-1, keepdims=True) + 10e-10)


def RELU(x: np.ndarray):
    return np.where(x > np.zeros(x.shape), x, np.zeros(x.shape))


def LRELU(x: np.ndarray):
    delta = 10e-4
    return np.where(x > np.zeros(x.shape), x, delta * x)


def derivate(func):
    if func.__name__ == "sigmoid":

        def func(x):
            return sigmoid(x) * (1 - sigmoid(x))

        return func

    elif func.__name__ == "RELU":

        def func(x):
            return np.where(x > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(x):
            delta = 10e-4
            return np.where(x > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)
