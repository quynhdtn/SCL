from sklearn import linear_model

from scipy.optimize import minimize

__author__ = 'quynhdo'
from scipy import linalg
import numpy as np
# Implements Structural Corresponding Learning Algorithm
# Linear version


def modified_huber_loss(W, X, y):
    p =  np.dot(W[0:-1],X.T) + W[-1]
    z = p * y
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.] = 0
    return np.sum(loss)

def modified_huber_dloss(W, X, y):
    p = np.dot(W[0:-1],X.T) + W[-1]
    z = p * y

    gb = -4.0 * y

    gb[z >= -1] = 2.0 * (1.0 - z[z >= -1]) * -y[z >= -1]
    gb[z >= 1.] = 0

    g = (-4.0 * y)[:,None] * X

    g[z >= -1] = (2.0 * (1.0 - z[z >= -1]) * -y[z >= -1])[:,None]* X
    g[z >= 1.] = 0

    g= np.concatenate((np.sum(g, axis=0), np.asarray([np.sum(gb)])))

    return g



def predictor(X, Y):
    W = np.zeros(X.shape[1]+1)

    res = minimize(modified_huber_loss, W, args=(X,Y), method="Newton-CG", jac=modified_huber_dloss)
    return res.x


def SCL_apply( X, pivot_feas, Y_trains, h, masked_feas=None):
    mask = np.asarray(pivot_feas)

    rs = []
    for i in range(np.sum(pivot_feas)):
        Xi = None
        if masked_feas is not None:
            Xi = X * Y_trains[i].T
        else:
            Xi = X * mask
        w = predictor(Xi, Y_trains[i])

        rs.append(w.T)
    rs = np.asarray(rs)
    rs = rs.reshape(rs.shape[0], rs.shape[1])
    rs = rs.T
    u, s, v = linalg.svd(rs, full_matrices=True)

    return u[0:h, :].T

class SCL :
    def __init__(self):
        self.all_W = []

    def add_pivot_predictor(self, X_train, Y_train):
        self.all_W.append(predictor(X_train, Y_train).T)


    def get_h(self, h):
        rs = self.all_W
        rs = np.asarray(rs)
        rs = rs.reshape(rs.shape[0], rs.shape[1])
        rs = rs.T
        u, s, v = linalg.svd(rs, full_matrices=True)

        return u[0:h, :].T


if __name__ == "__main__":

    '''
    test case
    "the girl"
    "the man"
    "girl man"
    feature = x, x[-1]
    bag of word "the girl man"
    pivot features: x = girl, x =man => indexes = 1,2
    '''
    '''

    def f(x, y):
        return x * x - 3 + y
    def g(x,y):
        return 2*x

    def main():
        x0 = .1
        y = 1
        res = minimize(f, x0, args=(y,), method="Newton-CG", jac=g)
        print(res)


    X = np.asarray( [[1 , 0, 0, 0 , 0,  0,1], [0,1,0,1,0,0,0], [1,0 ,0, 0,0,0,1], [0,0,1,1,0,0,0],[0,1,0,0,0,0,1] , [0,0,1,0,1,0,1]] )

    print (X.shape)
    y = np.ones(X.shape[0])
    w = np.zeros (X.shape[1])

    print(y.shape)

    W = predictor(X, y)
    print (W)

    p = np.sum(W[0:-1] * X, axis=1) + W[-1]
    print (p)


    from sklearn.datasets import load_iris
    data = load_iris()
    print(data)
    labels= data.target
    print (labels)
    labels[labels>0]=1
    labels[labels ==0] = -1
    print (labels.shape)
    data = np.asarray(data.data)
    print (data.shape)



    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size = 0.65, random_state = 42)

    print (y_test)

    W = predictor(X_train, y_train)
    print(W)

    p = np.sum(W[0:-1] * X_test, axis=1) + W[-1]
    print(p)

    p[p<0]=-1
    p[p>=0]=1
    print (p)
    from sklearn.metrics import f1_score
    print (f1_score(y_test, p))

    mdl= linear_model.LogisticRegression()
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_test)
    print(f1_score(y_test, pred))
    '''
    X = np.asarray([[1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0]])
    print(X)
    pivots = [0, 1, 1, 0, 0, 0]

    Y = [np.asarray([0, 1, 0, 0, 1, 0]), np.asarray([0, 0, 0, 1, 0, 1])]

    print(SCL_apply(X, pivots, Y, 2))