import numpy as np
from scipy.optimize import minimize
import time

class ContentBasedFiltering:
    y = np.array([])
    r = np.array([])
    X = np.array([])
    theta = np.array([])
    lastTrainDuration = ''
    currentCost = 0

    def cost(self, theta, X, y, r, nu, nm, nf):
        theta = np.reshape(theta, (nu, nf))
        return sum(sum(((np.matmul(X, theta.transpose()) - y) * r) ** 2)) / 2

    def grad(self, theta, X, y, r, nu, nm, nf):
        theta = np.reshape(theta, (nu, nf))
        return np.matmul(((np.matmul(X, theta.transpose()) - y) * r).transpose(), X).flatten()

    def iteration_callback(self, param):
        print('.', end='', flush=True)

    def train(self, y, r, X, theta):
        self.y = y
        self.r = r
        self.X = X
        self.theta = theta
        nm = y.shape[0]
        nu = y.shape[1]
        nf = X.shape[1]

        start_time = time.time()
        res = minimize(self.cost, theta.flatten(), args=(X, y, r, nu, nm, nf), method='CG', jac=self.grad, callback=self.iteration_callback, options={'disp': False, 'maxiter': 100.0})
        end_time = time.time()
        self.theta = np.reshape(res.x, (nu, nf))
        self.currentCost = self.cost(self.theta, self.X, self.y, self.r, nu, nm, nf)
        self.lastTrainDuration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    def predictAll(self):
        return np.matmul(self.X, self.theta.transpose())