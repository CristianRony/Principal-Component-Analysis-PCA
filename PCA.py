#Alumno: Cristian Rony Yaguno Mamani
#laboratoria de IA
#Tema PCA
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.mlab import PCA as mlabPCA

SiO2=np.array([50.18,50.67,50.78,50.94,49.67,50.73,49.91,50.93,50.08,51.74,50.31,50.47,50.75,49.98,51.02,50.85,50.07,50.99,49.72,57.43,49.88,49.74,50.26,50.3,51.15,50.92,50.85,49.99,50.15,51.44,51.44,51.09,51.1,51.65,51.04])
TiO2=np.array([1.42,1.36,1.57,1.53,1.19,2.38,1.05,1.4,1.13,1.58,1.52,1.29,1.32,1.32,1.95,2.04,1.93,1.7,1.94,1.9,1.24,1.24,1.48,1.6,1.41,1.48,1.52,1.66,1.28,1.65,1.65,0.8,2.32,1.47,2.02])
Al2O3=np.array([15.53,15.34,15.01,14.82,16.57,13.27,16.65,15.22,15.09,14.55,14.29,16.72,15.47,14.84,14.22,14.17,14.02,14.56,13.9,13.51,15.83,16.25,15.26,14.89,14.84,14.97,15.06,15.58,15.24,13.8,13.8,14.87,13.62,15.72,14.2])
Cr2O3=np.array([0.01,0,0,0.07,0.03,0.01,0,0.01,0,0,0.1,0.06,0.04,0.07,0.04,0.01,0.14,0.04,0.03,0.08,0.05,0.04,0,0,0.05,0.06,0,0.03,0.04,0,0,0,0,0.02,0.04])
FeO=np.array([9.3,8.79,9.7,9.71,8.26,12.88,7.89,9.26,9.12,10.47,10.53,8.32,8.85,9.46,11.28,11.34,10.98,10.55,11.41,11.54,8.62,8.51,9.15,9.84,9.48,9.62,9.11,9.64,8.64,1.58,11.58,0.54,12.17,8.92,11.01])
NiO=np.array([0.02,0,0.02,0.05,0,0.02,0,0,0,0,0.02,0.01,0.02,0.01,0.05,0.02,0.01,0.1,0.02,0,0,0.07,0,0.04,0.03,0,0,0,0.03,0,0,0.01,0,0,0.01])
MnO=np.array([0.06,0.08,0.23,0.18,0.14,0.29,0.03,0,0.09,0.16,0.09,0.16,0.16,0.22,0.25,9.05,0.22,0.25,0.1,0.09,0.02,0.19,0.13,0.24,0.14,0.150,0.14,0.2,0.02,0.22,0.22,0.12,0.21,0.13,0.11])
MgO=np.array([7.99,7.9,7.58,7.69,8.77,6.8,8.97,7.53,8.27,7.1,7.06,8.78,8.13,7.91,7.04,6.84,6.93,7.49,7.54,2.83,8.48,8.5,7.67,7.23,7.96,7.72,7.92,7.68,7.94,6.72,6.72,8.58,6.44,7.97,7.15])
CaO=np.array([11.93,12.42,11.41,11.39,11.39,10.33,12.45,12.16,12.65,10.82,11.51,11.32,11.73,12.53,10.4,10.91,11.47,11.32,11.36,6.83,12.25,11.61,11.69,12.04,11.4,11.19,11.72,11.16,12.69,10.8,10.8,12.91,10.44,10.88,10.57])
Na2O=np.array([2.83,2.95,2.93,2.93,2.83,2.95,2.53,2.93,2.7,3.22,3.26,2.83,2.86,2.91,2.9,2.86,3.1,3,2.8,4.5,2.82,2.91,3.17,2.33,2.72,2.69,2.8,3.2,2.59,2.59,2.59,2.04,2.93,2.3,2.92])
K2O=np.array([0.17,0.09,0.13,0.12,0.05,0.08,0.04,0.14,0.07,0.26,0.23,0.15,0.08,0.17,0.13,0.17,0.11,0.15,0.09,0.47,0.04,0.03,0.15,0.19,0.05,0.08,0.08,0.18,0.06,0.21,0.21,0.07,0.11,0.05,0.13])

X = np.array(list(zip(SiO2, TiO2, Al2O3, Cr2O3, FeO, NiO, MnO, MgO, CaO, Na2O, K2O))).reshape(len(SiO2), 11)
plt.plot(SiO2, MgO,'ro')
plt.ylabel('MgO')
plt.xlabel('SiO2')
plt.show
print('Analis de datos principales')
print('========================================================')
print(X)
print('========================================================')
M=mean(X.T, axis=1)
print('Promedio en cada columna de datos: ')
print(M)
# columnas centrales restando medios de columna
C=X-M
print('Columnas centrales restando medios de columna')
print(C)
print('========================================================')
#covarianza
V = cov(C.T)
print('Matriz de Covarianza')
print(V)
print('========================================================')
# descomposición propia de la matriz de covarianza
values, vectors = eig(V)
print('vectores')
print(vectors)
print('values')
print(values)
# datos del proyecto
print('---------------')
P = vectors.T.dot(C.T)
print(P.T)
print('========================================================')
#Análisis de componentes principales reutilizables

# crear la instancia de PCA
pca = PCA(2)
# ajuste en los datos
pca.fit(X)
# acceso a valores y vectores
print('PCA componentes')
print(pca.components_)
print('PCA varianza')
print(pca.explained_variance_)
# transformar datos
print('Transformacion de Datos')
B = pca.transform(X)
print(B)
print('========================================================')
print('las relaciones entre MgO y SiO2 en los componentes principales')
Y = np.array(list(zip(SiO2, MgO))).reshape(len(SiO2), 2)
pca = PCA(2)
# ajuste en los datos
pca.fit(Y)
# acceso a valores y vectores
print('PCA componentes')
print(pca.components_)
print('PCA varianza')
print(pca.explained_variance_)
print('========================================================')
print('========================================================')
print('========================================================')




