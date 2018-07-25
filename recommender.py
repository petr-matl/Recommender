import numpy as np
import pandas as pd
from MoviesDataset import MoviesDataset
from ColaborativeFiltering import ColaborativeFiltering
from ContentBasedFiltering import ContentBasedFiltering


dataset = 'real'
data = MoviesDataset(dataset)

modelColaborative = ColaborativeFiltering()
modelContentBased = ContentBasedFiltering()

print('Content-Based Filtering')
y, r, X, theta, nm, nu, nf = data.getData('content-based')
modelContentBased.train(y, r, X, theta)
print('\n')
pd.DataFrame(modelContentBased.theta).to_csv(dataset + "_content-based_theta.csv", header=None, index=None)

print('Colaborative Filtering')
y, r, X, theta, nm, nu, nf = data.getData('colaborative')
modelColaborative.train(y, r, X, theta)
print('\n')
pd.DataFrame(modelColaborative.theta).to_csv(dataset + "_colaborative_theta.csv", header=None, index=None)
pd.DataFrame(modelColaborative.X).to_csv(dataset + "_colaborative_X.csv", header=None, index=None)

print('ContentBased cost: {0:,.4f}'.format(modelContentBased.currentCost).replace(',', ' '))
print('Colaborative cost: {0:,.4f}'.format(modelColaborative.currentCost).replace(',', ' '))

print('ContentBased duration: {0}'.format(modelContentBased.lastTrainDuration))
print('Colaborative duration: {0}'.format(modelColaborative.lastTrainDuration))

#print('ContentBased prediction:')
#print(predict_ContentBased)
#print('Colaborative prediction:')
#print(predict_Colaborative)



'''
index = 1
def callbackContentBased(param):
    print('.', end='', flush=True)
    global index
    #print("{0}: Time {1}, Cost {2}".format(index, datetime.now().time(), costContentBased(param, X, y, r, nu, nm, nf)))
    f = open("logFile.txt","a+")
    f.write("{0}: Time {1}, CostContentBased {2}\n".format(index, datetime.now().time(), costContentBased(param, X, y, r, nu, nm, nf)))
    f.close()
    index += 1
    #pd.DataFrame(np.reshape(theta, (nu, nf))).to_csv("content-based_theta.csv", header=None, index=None)

def callbackColaborative(param):
    print('.', end='', flush  =True)
    global index
    #print("{0}: Time {1}, Cost {2:,.4f}".format(index, datetime.now().time(), costColaborative(param, y, r, nu, nm, nf)))
    f = open("logFile.txt","a+")
    f.write("{0}: Time {1}, CostColaborative {2}\n".format(index, datetime.now().time(), costColaborative(param, y, r, nu, nm, nf)))
    f.close()
    index += 1
    #pd.DataFrame(np.reshape(param[0:nu*nf], (nu, nf))).to_csv("colaborative_theta.csv", header=None, index=None)
    #pd.DataFrame(np.reshape(param[nu*nf:nu*nf+nm*nf], (nm, nf))).to_csv("colaborative_X.csv", header=None, index=None)

try:
    theta = pd.read_csv(dataset + "_content-based_theta.csv", header=None).values
    print(dataset + "_content-based_theta.csv was successfully loaded!")
except FileNotFoundError:
    print(dataset + "_content-based_theta.csv was not Found!")

try:
    theta = pd.read_csv(dataset + "_colaborative_theta.csv", header=None).values
    X = pd.read_csv(dataset + "_colaborative_X.csv", header=None).values
    print(dataset + "_colaborative_theta.csv was successfully loaded!")
    print(dataset + "_colaborative_X.csv was successfully loaded!")
except FileNotFoundError:
    print(dataset + "_colaborative_theta.csv was not Found!")
    print(dataset + "_colaborative_X.csv was not Found!")

'''