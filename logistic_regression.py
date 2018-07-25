import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

def cost(theta, X, y):
    m = len(y)
    tmp_mat = expit(np.matmul(X, theta))
    return (1 / m) * sum(-y * np.log(tmp_mat) - (1 - y) * np.log(1 - tmp_mat))

def cost_grad(theta, X, y):
    m = len(y)
    tmp_mat = expit(np.matmul(X, theta))
    return (1 / m) * np.matmul(X.transpose(), (tmp_mat - y))

data = pd.read_csv('ex2data1.txt', sep=',', header=None)
X = data.loc[:, 0:1]
X = pd.concat([pd.Series(np.ones(len(X))), X], axis=1)
y = data.loc[:, 2]
theta = np.zeros(X.shape[1])

print(cost(theta, X, y))
print(cost_grad(theta, X, y))

res = minimize(cost, theta, args=(X, y), method='BFGS', jac=cost_grad, options={'disp': True})
print(res.x)