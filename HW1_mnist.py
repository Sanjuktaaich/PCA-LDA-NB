#Author: Sanjukta Aich
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy import linalg as sp 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def PCA_train(data):
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
	print(x)
	m1=np.mean(data1,axis=0)
	m2=np.mean(data2,axis=0)
	m1=m1.reshape(x,1)
	m2=m2.reshape(x,1)
	data=np.concatenate((data1,data2),axis=0)
	m=np.mean(data,axis=0)
	m=m.reshape(x,1)
	Sw=np.cov(data1.T) + np.cov(data2.T)
	print(Sw.shape)
	Sb=np.dot((m1-m2),((m1-m2).T))
	print(Sw.shape)
	e,v=np.linalg.eig((np.linalg.pinv(Sw)).dot(Sb))
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
	m1=(np.mat(data1.mean(axis=1))).reshape(784,1)
	m2=(np.mat(data2.mean(axis=1))).reshape(784,1)
	s1=(np.std(data1,axis=1)).reshape(784,1)
	s2=(np.std(data2,axis=1)).reshape(784,1)
	pred=np.zeros((x.shape[1]))
	for i in range(x.shape[1]):
		p1=0
		p2=0
		for j in range(x.shape[0]):
			if s1[j]!=0:
				p1=p1*(1/(np.sqrt(2*3.14)*s1[j]))*np.exp(-((x[j,i]-m1[j])*(x[j,i]-m1[j])/(2*(s1[j]*s1[j]))))
			if s2[j]!=0:
				p2=p2*(1/(np.sqrt(2*3.14)*s2[j]))*np.exp(-((x[j,i]-m2[j])*(x[j,i]-m2[j])/(2*(s2[j]*s2[j]))))
		if(p2>p1):
			pred[i]=1
		else:
			pred[i]=0
	return pred

#MNIST dataset

mnist_train = (np.genfromtxt('MNIST/train.csv', delimiter=','))
print(mnist_train.shape)
mnist_test = (np.genfromtxt('MNIST/test.csv', delimiter=','))
print(mnist_test.shape)
i0=np.mat(np.where(mnist_train[784,:]==0)).T
i1=np.mat(np.where(mnist_train[784,:]==1)).T
i3=np.mat(np.where(mnist_train[784,:]==3)).T
i5=np.mat(np.where(mnist_train[784,:]==5)).T
train0=mnist_train[:784,i0].reshape(784,i0.shape[0])
train1=mnist_train[:784,i1].reshape(784,i1.shape[0])
train3=mnist_train[:784,i3].reshape(784,i3.shape[0])
train5=mnist_train[:784,i5].reshape(784,i5.shape[0])
i0=np.mat(np.where(mnist_test[784,:]==0)).T
i1=np.mat(np.where(mnist_test[784,:]==1)).T
i3=np.mat(np.where(mnist_test[784,:]==3)).T
i5=np.mat(np.where(mnist_test[784,:]==5)).T
test0=mnist_test[:784,i0].reshape(784,i0.shape[0])
test1=mnist_test[:784,i1].reshape(784,i1.shape[0])
test3=mnist_test[:784,i3].reshape(784,i3.shape[0])
test5=mnist_test[:784,i5].reshape(784,i5.shape[0])
l=i0.shape[0]+i1.shape[0]
l1=i3.shape[0]+i5.shape[0]

e,v=PCA_train(np.concatenate((train0,train1),axis=1))
pca=PCA_test(np.concatenate((test0,test1),axis=1),v,2)
rec1=v[:,:100]
rec1=rec1.dot(rec1.T)
rec1=rec1.dot(train0[:,0])
rec1=rec1.reshape(28,28)
print(v[:,0])
plt.figure(1, figsize=(8,6))
plt.clf()
plt.scatter(pca[i0,0],pca[i0,1],c='blue',label='Number 0')
plt.scatter(pca[i1,0],pca[i1,1],c='red',label='Number 1')
plt.legend(loc="upper right")
plt.xticks(())
plt.yticks(())
e,v=PCA_train(np.concatenate((train3,train5),axis=1))
pca=PCA_test(np.concatenate((test3,test5),axis=1),v,2)
plt.figure(2, figsize=(8,6))
plt.clf()
plt.scatter(pca[(i3-l),0],pca[(i3-l),1],c='blue',label='Number 3')
plt.scatter(pca[(i5-l),0],pca[(i5-l),1],c='red',label='Number 5')
plt.legend(loc="upper right")
ep=np.cumsum((e))/np.sum(e)
r=np.sum(e[2:])/np.sum(e)
print(r)
plt.figure(3, figsize=(8,6))
plt.clf()
plt.plot(ep,label='MNIST- PCA')

e,v=LDA_train(train0.T,train1.T)
test=np.concatenate((test0,test1),axis=1)
lda=LDA_test(test.T,v,2)
rec2=v[:,:300]
rec2=rec2.dot(rec2.T)
rec2=rec2.dot(train0[:,0])
rec2=rec2.reshape(28,28)
print(v[:,0])
rec2=v[:,0].reshape(28,28)
rec2=rec2.dot(rec2.T)
plt.figure(4, figsize=(8,6))
plt.clf()
plt.scatter(lda[i0,0],lda[i0,1],c='blue',label='Number 0')
plt.scatter(lda[i1,0],lda[i1,1],c='red',label='Number 1')
plt.legend(loc="upper right")
plt.xticks(())
plt.yticks(())
m1=np.mean((train0.T).dot(v[:,0]),axis=0)
m2=np.mean((train1.T).dot(v[:,0]),axis=0)
m=(m1+m2)/2
test=np.concatenate((test0,test1),axis=1)
y_true=np.zeros((test.shape[1]))
t=test.shape[1]-test0.shape[1]
y_true[t:]+=1
y_pred=LDA_classify(m,test.T,v[:,0],m1,m2)
cm1=confusion_matrix(y_true, y_pred)
print(cm1)
target_names = ['0', '1']
print(classification_report(y_true, y_pred, target_names=target_names))

e,v=LDA_train(train3.T,train5.T)
test=np.concatenate((test3,test5),axis=1)
lda=LDA_test(test.T,v,2)
plt.figure(4, figsize=(8,6))
plt.clf()
plt.scatter(lda[i3-l,0],lda[i3-l,1],c='blue',label='Number 3')
plt.scatter(lda[i5-l,0],lda[i5-l,1],c='red',label='Number 5')
plt.legend(loc="upper right")
m1=np.mean((train3.T).dot(v[:,0]),axis=0)
m2=np.mean((train5.T).dot(v[:,0]),axis=0)
m=(m1+m2)/2
test=np.concatenate((test3,test5),axis=1)
y_true=np.zeros((test.shape[1]))
t=test.shape[1]-test3.shape[1]
y_true[t:]+=1
y_pred=LDA_classify(m,test.T,v[:,0],m1,m2)
cm2=confusion_matrix(y_true, y_pred)
print(cm2)
target_names = ['3', '5']
print(classification_report(y_true, y_pred, target_names=target_names))
ep=np.cumsum((e))/np.sum(e)
r=np.sum(e[2:])/np.sum(e)
print(r)
plt.figure(5, figsize=(8,6))
plt.clf()
plt.plot(ep,label='MNIST- LDA')
#plt.show()

e,v=LDA_train(train0.T,train1.T)
p=LDA_test(np.concatenate((train0,train1),axis=1).T,v,784)
result=Bayes(p[:train0.shape[1],:].T,p[train0.shape[1]:,:].T,np.concatenate((test0,test1),axis=1))
e,v=LDA_train(train3.T,train5.T)
p=LDA_test(np.concatenate((train3,train5),axis=1).T,v,784)
result=Bayes(p[:train3.shape[1],:].T,p[train3.shape[1]:,:].T,np.concatenate((test0,test1),axis=1))

plt.figure(6, figsize=(8,6))
plt.imshow(cm1)
plt.figure(7, figsize=(8,6))
plt.imshow(cm2)
plt.figure(8, figsize=(8,6))
plt.imshow(np.absolute(rec1))
plt.figure(9, figsize=(8,6))
plt.imshow(np.absolute(rec2))
o=train0[:,0].reshape(28,28)
plt.figure(10, figsize=(8,6))
plt.imshow(np.absolute(o-rec1))
plt.figure(11, figsize=(8,6))
plt.imshow(np.absolute(o-rec2))
plt.show()