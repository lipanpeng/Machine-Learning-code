import numpy as np
import matplotlib.pyplot as plt
# import random
# import tensorflow as tf


def get_data():
    data = np.array([[0.697, 0.460, 1],
                     [0.774, 0.376, 1],
                     [0.634, 0.264, 1],
                     [0.608, 0.318, 1],
                     [0.556, 0.215, 1],
                     [0.403, 0.237, 1],
                     [0.481, 0.149, 1],
                     [0.437, 0.211, 1],
                     [0.666, 0.091, 0],
                     [0.243, 0.267, 0],
                     [0.245, 0.057, 0],
                     [0.343, 0.099, 0],
                     [0.639, 0.161, 0],
                     [0.657, 0.198, 0],
                     [0.360, 0.370, 0],
                     [0.593, 0.042, 0],
                     [0.719, 0.103, 0]])
    x = data[:, :2].astype(np.float32)
    b = np.ones((x.shape[0], 1))
    x = np.concatenate([x, b], axis=1)
    y = data[:, 2].astype(np.float32)
    y = np.expand_dims(y, axis=1)
    # print('type of x is {}'.format(type(x)))
    # print('shape of x is {}'.format(x.shape))
    # print('x is {}'.format(x))
    print('shape of y is {}'.format(y.shape))

    return x, y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gradDescent(x, y, lr=0.05, iter=150):
    n = x.shape[0]

    # w = np.random.normal(0, 0.002, x.shape[1])
    # # w = np.ones(x.shape[1])
    # w = np.reshape(w, [1, x.shape[1]])
    # print('w is {}'.format(w))
    # b = np.zeros([1, ])
    # beta = np.ones((1, 3)) * 0.1
    beta = np.random.normal(0, 0.002, x.shape[1])
    beta = np.reshape(beta, (1, x.shape[1]))
    z = x.dot(beta.T)
    print('before optimal, beta is {}'.format(beta))
    l = np.sum((-y * z) + np.log(1 + np.exp(z)))
    print('before optimal, l is {}'.format(l))
    for i in range(iter):
        p1 = sigmoid(z)
        # print('p1 is {}'.format(p1))
        grads = -np.sum(x * (y - p1), axis=0, keepdims=True)
        beta -= grads * lr
        z = x.dot(beta.T)

    l = np.sum(-y * z + np.log(1 + np.exp(z)))
    print('after optimal, l is {}'.format(l))
    print('after optimal, beta is {}'.format(beta))
    return beta



if __name__ == '__main__':
    x, y = get_data()
    for i in range(x[:, 0].shape[0]):
        if y[i, 0] == 0:
            plt.plot(x[:, 0][i], x[:, 1][i], 'r+')

        else:
            plt.plot(x[:, 0][i], x[:, 1][i], 'bo')
    beta = gradDescent(x, y)
    grad_descent_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    grad_descent_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [grad_descent_left, grad_descent_right], 'y-')

    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title("LR")
    plt.show()


