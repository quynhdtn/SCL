import numpy as np

x= np.asarray([[1,2],[3,4],[5,6]])
t = np.asarray([2,3,4])
print (t[:,None] * x)