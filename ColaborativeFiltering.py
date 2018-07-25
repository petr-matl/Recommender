import numpy as np
from scipy.optimize import minimize
import time

class ColaborativeFiltering:
    y = np.array([])
    r = np.array([])
    X = np.array([])
    theta = np.array([])
    lastTrainDuration = ''
    currentCost = 0

    def cost(self, param, y, r, nu, nm, nf):
        theta = np.reshape(param[0:nu*nf], (nu, nf))
        X = np.reshape(param[nu*nf:nu*nf+nm*nf], (nm, nf))
        return sum(sum(((np.matmul(X, theta.transpose()) - y) * r) ** 2)) / 2

    def grad(self, param, y, r, nu, nm, nf):
        theta = np.reshape(param[0:nu*nf], (nu, nf))
        X = np.reshape(param[nu*nf:nu*nf+nm*nf], (nm, nf))
        theta_grad = np.matmul(((np.matmul(X, theta.transpose()) - y) * r).transpose(), X)
        X_grad = np.matmul(((np.matmul(X, theta.transpose()) - y) * r), theta)
        return np.append(theta_grad.flatten(), X_grad.flatten())

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
        res = minimize(self.cost, np.hstack((theta.flatten(), X.flatten())), args=(y, r, nu, nm, nf), method='CG', jac=self.grad, callback=self.iteration_callback, options={'disp': False, 'maxiter': 100.0})
        end_time = time.time()
        self.theta = np.reshape(res.x[0:nu*nf], (nu, nf))
        self.X = np.reshape(res.x[nu*nf:nu*nf+nm*nf], (nm, nf))
        self.currentCost = self.cost(res.x, self.y, self.r, nu, nm, nf)
        self.lastTrainDuration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    def predictAll(self):
        return np.matmul(self.X, self.theta.transpose())
