#!/usr/bin/python

print __doc__


# Code source: Gael Varoqueux
# Modified for Documentation merge by Jaques Grobler
# License: BSD

import numpy as np
from sklearn import neighbors,linear_model,decomposition,cross_validation, preprocessing,metrics
from sklearn.neighbors.nearest_centroid import NearestCentroid

import sys
import gdbm
import cPickle
# import some data to play with


db1 = gdbm.open(sys.argv[1],"r")
if len(sys.argv) >= 3: 
 db2 = gdbm.open(sys.argv[2],"r")
else:
 db2 = None 
if len(sys.argv) == 4:
 db3 = gdbm.open(sys.argv[3],"r")
else:
 db3 = None
f = open("classes.txt")
cl = cPickle.load(f)
f.close()

name_arr = np.array(cl.keys())
data1  = np.fromstring(db1[name_arr[0]],sep=' ')
if db2 != None:
 data2 =  np.fromstring(db2[name_arr[0]],sep=' ')
else:
 data2 = np.array([])
if db3 != None:
 data3 =  np.fromstring(db3[name_arr[0]],sep=' ')
else:
 data3 = np.array([])
data = np.hstack((data1,data2[1:],data3))
for nome in name_arr[1:]:
 data1  = np.fromstring(db1[nome],sep=' ')
 data2 = np.array([])
 data3 = np.array([])
 if db2 != None:
  data2 =  np.fromstring(db2[nome],sep=' ')
 if db3 != None:
  data3 =  np.fromstring(db3[nome],sep=' ')
 aux = np.hstack((data1,data2[1:],data3))
 data = np.vstack((data,aux))
# Aqui os dados ja estao lidos em uma matriz X
# Em Y esta as classes de cada uma das observacoes 
X = preprocessing.scale(data[:,1:],copy=False)
#X = data[:,1:]
Y = data[:,0]
#print X,Y
# Entradas : combinacao linear das variaveis
# Numero de objetos e de caracteristicas observadas
Nobj = data.shape[0]
Nfe = data.shape[1]
it = cross_validation.StratifiedShuffleSplit(Y,5000,test_size = 0.2)
kl1 = neighbors.NearestCentroid(metric="l1")
kl2 = linear_model.LogisticRegression()
print X.shape
pca = decomposition.KernelPCA(n_components= 20,eigen_solver = 'arpack');
X = pca.fit_transform(X)
print X.shape
score1 = cross_validation.cross_val_score(kl1, X, Y, cv=it,score_func=metrics.f1_score)
score2 = cross_validation.cross_val_score(kl2, X, Y, cv=it,score_func=metrics.f1_score)

print score1.mean(),score1.std()
print score2.mean(),score2.std()
      
