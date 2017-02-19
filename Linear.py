import numpy as np
def modified_huber_loss(W, X, y):
    p = np.sum( np.W[0:-1] * X, axis=1) + W[-1]
    z = p * y
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.] = 0
    return np.sum(loss)

def modified_huber_dloss(W, X, y):
    p =np.sum( W[0:-1] * X , axis=1)+ W[-1]
    z = p * y

    gb = -4.0 * y

    gb[z >= -1] = 2.0 * (1.0 - z[z >= -1]) * -y[z >= -1]
    gb[z >= 1.] = 0
    print (gb.shape)

    g = -4.0 * y * X

    g[z >= -1] = 2.0 * (1.0 - z[z >= -1]) * -y[z >= -1]* X
    g[z >= 1.] = 0
    g= np.sum(g, axis=1)
    g= np.concatenate((g, np.asarray([np.sum(gb)])))
    print (g)
    return g

class BinaryClassification:
    def __init__(self):
        pass


