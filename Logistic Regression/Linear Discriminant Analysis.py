import numpy as np
import matplotlib.pyplot as plt
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

data = np.array([i[:-1] for i in data])
x0 = data[:8]
x1 = data[8:]

mu0 = np.mean(x0, axis=0).reshape((-1, 1))
mu1 = np.mean(x1, axis=0).reshape((-1, 1))

cov0 = np.cov(x0, rowvar=False)
cov1 = np.cov(x1, rowvar=False)

s_w = np.mat(cov0 + cov1)

omiga = s_w.I * (mu0 - mu1)

plt.scatter(x0[:, 0], x0[:, 1], c='b', label='+', marker='+')
plt.scatter(x1[:, 0], x1[:, 1], c='r', label='-', marker='_')
plt.plot([0, 1], [0, -omiga[0] / omiga[1]], label='y')
plt.xlabel('密度', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('含糖量', fontproperties='SimHei', fontsize=15, color='green')
plt.title('线性判别分析', fontproperties='SimHei', fontsize=25)
plt.legend()
plt.show()


