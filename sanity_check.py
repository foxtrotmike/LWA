#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:01:12 2017

@author: amina
"""
"""creates toy example and test LWA on it"""
from example import Example
from classifiers import linclass_rej
import numpy as np
from random import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt #importing plotting module
from plotit import plotit

transformer = RBFSampler(gamma=0.1, n_components=200)
transformer = PolynomialFeatures(2)

def loss_rej(y, h, r=1, c=0):
    if type(r)!=type(1):
        r=np.array(r)
    l_total=np.mean(np.sum(((y*h)<=0)*(r>0)+c*(r<=0)))
    l_rejection=np.mean(np.sum(c*(r<=0)))
    return l_total, l_rejection


def create_ex_gauss(m1=1.0, m2=-1.0, sd=1.4, n=500):        #create example gaussian
    Xp =sd* np.random.randn(n,2)+m1
    Xn =sd* np.random.randn(n,2)+m2
    X = np.vstack((Xp,Xn))
    m = np.mean(X,axis=0)
    s = np.std(X,axis=0)
#    Xp = (Xp-m)/s
#    Xn = (Xn-m)/s
    data=[]
    Xp2=transformer.fit_transform(Xp)
    Xn2=transformer.fit_transform(Xn)
    for i in range(len(Xp)):
        ex=Example()
        ex.features_u=Xp2[i]
        ex.raw_features=Xp[i]
        ex.features_w=Xp2[i]
        ex.label=1.0
        ex.gamma=1.0#/len(Xp)
        data.append(ex)
    for i in range(len(Xn)):
        ex=Example()
        ex.features_u=Xn2[i]
        ex.raw_features=Xn[i]
        ex.features_w=Xn2[i]
        ex.label=-1.0
        ex.gamma=1.0#/len(Xn)
        data.append(ex)
    shuffle(data)
    return data



if __name__ == '__main__':
    plt.close('all')
    
    examples=create_ex_gauss(m1=1.0, m2=2.0, sd=1.3, n=100)
    
#    classifier=linclass_rej(epochs=300, Lambda_u=0.1, Lambda_w=0.1, alpha=1.0, c=0.40)
#    classifier.train(examples)
    
    classifier3=linclass_rej(epochs=500, Lambda_u=0.1, Lambda_w=0.01, alpha=1.0, c=0.4)
    classifier3.train3(examples)
    
#    plt.plot(classifier.losses); 
    plt.plot(classifier3.losses)
    
    test=examples#create_ex_gauss(m1=1.0, m2=2.0, sd=1.0, n=100)     
    X = np.array([e.raw_features for e in test])
    Y = np.array([e.label for e in test])
    plt.figure()
#    plotit(X,Y,clf=classifier.classify, transform = transformer.fit_transform, conts =[-1,0,1] )
#    plotit(X,Y,clf=classifier.reject, transform = transformer.fit_transform, conts =[-1,0,1] )
    
    plotit(X,Y,clf=classifier3.reject, transform = transformer.fit_transform, conts =[0], ccolors = ['g'], hold = True )
    plotit(X,Y,clf=classifier3.classify, transform = transformer.fit_transform, conts =[-1,0,1])
    