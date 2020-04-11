from numpy import mean
import numpy as np
from sklearn import preprocessing
p=np.array([50.93,50.08,51.74,50.31,50.47,50.75,49.98,51.02,50.85,50.07])
y=np.array([1.42,1.36,1.57,1.53,1.19,2.38,1.05,1.4,1.13,1.58])
z=np.array([9.3,8.79,9.7,9.71,8.26,12.88,7.89,9.26,9.12,10.47])
X = np.array(list(zip(p,y,z))).reshape(len(p), 3)
x_normalized = preprocessing.normalize(X,norm= 'l2')
print(x_normalized)

M=mean(X.T, axis=1)
