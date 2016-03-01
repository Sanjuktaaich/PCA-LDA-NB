#Author: Sanjukta Aich
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
from scipy import linalg as sp 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def PCA_train(data):
	m,n=data.shape
	data_orig=data
	m=data.mean(axis=1)
	C=np.cov(data)
	e,v=sp.eigh(C)
	i=np.argsort(e)[::-1]
	e=e[i]
	v=v[:,i]
	return e,v

def PCA_test(data, v, dim):
	w=v[:,:dim]
	pca=(data.T).dot(w)
	return pca

def LDA_train(data1, data2):
	n1,x=data1.shape
	n2,x=data2.shape
	m1=np.mean(data1,axis=0)
	m2=np.mean(data2,axis=0)
	m1=m1.reshape(x,1)
	m2=m2.reshape(x,1)
	data=np.concatenate((data1,data2),axis=0)
	m=np.mean(data,axis=0)
	m=m.reshape(x,1)
	Sw=np.cov(data1.T) + np.cov(data2.T)
	Sb=np.dot((m1-m2),((m1-m2).T))
	e,v=np.linalg.eig((np.linalg.inv(Sw)).dot(Sb))
	i=np.argsort(np.absolute(e.real))[::-1]
	e=e[i].real
	v=v[:,i]
	return e,v

def LDA_test(data, v, dim):
	w=v[:,:dim]
	lda=data.dot(w)
	return lda

def LDA_classify(m,x,v,m1,m2):
	p=x.dot(v)
	y_pred=np.zeros((p.shape[0]))
	for i in range(p.shape[0]):
		if p[i]>m and m2>m1:
			y_pred[i]=1
		elif p[i]>m and m1>m2:
			y_pred[i]=0
		elif p[i]<m and m2>m1:
			y_pred[i]=0
		elif p[i]<m and m1>m2:
			y_pred[i]=1
	return y_pred

def Bayes(data1, data2, x):
	m1=np.mat(data1.mean(axis=1))
	m2=np.mat(data2.mean(axis=1))
	s1=np.std(data1,axis=1)
	s2=np.std(data2,axis=1)
	p1=1
	p2=1
	pred=np.zeros((x.shape[1]))
	for i in range(x.shape[1]):
		p1=1
		p2=1
		for j in range(x.shape[0]):
			p1=p1*(1/(np.sqrt(2*3.14)*s1[j]))*np.exp(-((x[j,i]-m1[j])*(x[j,i]-m1[j])/(2*(s1[j]*s1[j]))))
			p2=p2*(1/(np.sqrt(2*3.14)*s2[j]))*np.exp(-((x[j,i]-m2[j])*(x[j,i]-m2[j])/(2*(s2[j]*s2[j]))))
		if(p1>p2):
			pred[i]=0
		else:
			pred[i]=1
	return pred

#WINE dataset

f=open("WINE/wine.txt")
f.readline()
wine = np.mat(np.loadtxt(f, delimiter=","))
t=5
i1=random.sample(range(0,59),t)
i2=random.sample(range(59,130),t)
i=np.concatenate((i1,i2),axis=0)

e,v=PCA_train(wine[i,1:].T)
test=np.delete(wine[0:130,1:],i,axis=0)
pca=PCA_test(test.T,v,2)
plt.figure(1, figsize=(8,6))
print(v[:,0].real)
print(v[:,12].real)
plt.clf()
plt.figure(2, figsize=(8,6))
plt.scatter(pca[0:(59-t),0],pca[0:(59-t),1],c='blue',label='class 1')
plt.scatter(pca[(59-t):(130-t),0],pca[(59-t):(130-t),1],c='red',label='class 2')
plt.xticks(())
plt.yticks(())
ep=np.cumsum((e))/np.sum(e)
r=np.sum(e[2:])/np.sum(e)
print(r)
plt.figure(3, figsize=(8,6))
plt.clf()
plt.plot(ep,label='WINE- PCA')

e,v=LDA_train(wine[i1,1:],wine[i2,1:])
test1=np.delete(wine[0:59,1:],i1,axis=0)
test2=np.delete(wine[59:130,1:],i2,axis=0)
test=np.delete(wine[0:130,1:],i,axis=0)
lda=LDA_test(test,v,2)
m1=np.mean(wine[i1,1:].dot(v[:,0]),axis=0)
m2=np.mean(wine[i2,1:].dot(v[:,0]),axis=0)
m=(m1+m2)/2
y_true=np.zeros((130-(2*t)))
y_true[59-t:]+=1
y_pred=LDA_classify(m,test,v[:,0],m1,m2)
print(y_true)
print(y_pred)
cm1=confusion_matrix(y_true, y_pred)
print(cm1)

print(v[:,0].real)
print(v[:,12].real)
lda=(lda.real)
ep=np.cumsum((e))/np.sum(e)
r=np.sum(e[2:])/np.sum(e)
print(r)
plt.plot(ep,label='WINE- LDA')
plt.xlabel('Eigen vector number')
plt.ylabel('Cummulative sum of the eigen vectors')
plt.legend(loc="upper right")
plt.figure(4, figsize=(8,6))
plt.clf()
plt.scatter(lda[:(59-t),0],lda[:(59-t),1],c='blue',label='class 1')
plt.scatter(lda[(59-t):(130-t),0],lda[(59-t):(130-t),1],c='red',label='class 2')
plt.xticks(())
plt.yticks(())
plt.figure(5, figsize=(8,6))
plt.imshow(cm1)
#classify

train1=wine[i1,1:]
train2=wine[i2,1:]
print(test.shape)
y_pred=Bayes(train1.T,train2.T,test.T)
y_true=np.zeros((130-(2*t)))
y_true[59-t:]+=1
print(y_true)
print(y_pred)
cm2=confusion_matrix(y_true, y_pred)
print(cm2)
plt.figure(6, figsize=(8,6))
plt.imshow(cm2)
plt.show()
target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred, target_names=target_names))
        