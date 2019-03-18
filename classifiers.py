# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:59:45 2015

@author: amina
This module contains the class definitions for the Stochastic subgradient descent based large margin classifiers for Multiple Instance Learning
Following classifiers are implemented here:
linclass-Linear Classifier
linclass_rank- Ranking Implementation of Linear Classifier
llclass- Non Linear Classifier
llclass_rank- Ranking Implementation of Non Linear Classifier 
"""


from scipy.sparse import *
import random
import numpy as np
#from cv import *
from example import Example

from random import shuffle
#########################################################################################
class ClassifierBase:
    
  
    
    def __init__(self,*args, **kwargs):
        
        if len(args) and isinstance(args[0], self.__class__):            
            
            self.epochs=args[0].epochs
          
        else:
        
            if 'epochs' in kwargs:
                self.epochs=kwargs['epochs']
            else:
                self.epochs=100
           
        
    def train(self,bags,**kwargs):
        pass
    
    
    def classify(self,test_example):
        
       pass
    
    def reject(self,test_example):
        
        
        pass
    
    
    def predict(self, test_example):

        pass
   
    
#############################################################################################

class linclass_rej(ClassifierBase, object):  
    def __init__(self,*args, **kwargs):
        super(linclass_rej, self).__init__(*args, **kwargs)
        if len(args) and isinstance(args[0], self.__class__):
            self.Lambda_u=args[0].Lambda_u
            self.Lambda_w=args[0].Lambda_w      
            self.alpha=args[0].alpha
            self.c=args[0].c
        else:
                    
            if 'Lambda_u' in kwargs:
                self.Lambda_u=kwargs['Lambda_u']
            else:
                self.Lambda_u=0.00001
            if 'Lambda_w' in kwargs:
                self.Lambda_w=kwargs['Lambda_w']
            else:
                self.Lambda_w=0.00001
            if 'alpha' in kwargs:
                self.alpha=kwargs['alpha']
            else:
                self.alpha=1
            if 'c' in kwargs:
                self.c=kwargs['c']
            else:
                self.c=0.4
                
               
        self.w=None
        self.u=None
        
        
        self.beta=1/(1-2*self.c)
        
        
    def classify(self,test_example):
        
        w=self.w
        
        f=None
        e=Example()
        if type(test_example)==type(e):
            f=test_example.features_w
            
        else:
            f=test_example
        h=f.dot(w.T) 

        return h   
    
    def reject(self,test_example):
        
        
        u=self.u
        f=None
        e=Example()
        if type(test_example)==type(e):
            f=test_example.features_u
            
        else:
            f=test_example
        r=f.dot(u.T) 
        return r 
    
    
    def predict(self, test_example):

        if self.reject(test_example)<0:
            return 'r'
        else:
            
            score=self.classify(test_example)
            if score>0:
                return 1.0
            else:
                return -1.0    

    def train(self, examples,**kwargs): #with wbar
        
#        print "c=", self.c 
        siz1=np.shape(examples[0].features_w)[0]
        siz2=np.shape(examples[0].features_u)[0]
        
        w=np.array(np.zeros(siz1)) 
        wbar=np.array(np.zeros(siz1)) 
        
        u=np.array(np.zeros(siz2))
        ubar=np.array(np.zeros(siz2)) 
        
        self.w=wbar
        self.u=ubar

        epochs=self.epochs
        
        T=(len(examples))*epochs
        lambdaa_u=self.Lambda_u
        lambdaa_w=self.Lambda_w

        a=self.alpha
        b=self.beta
        c=self.c
        loss=np.zeros(epochs)
        curr_epoch=0
        
        for t in range(1,T+1):
            rho=2.0/(t+2.0)
            eeta_u=1.0/(lambdaa_u*t)
            eeta_w=1.0/(lambdaa_w*t)
            ex_chosen=examples[(t-1)%len(examples)]

            classify_score=self.classify(ex_chosen)
            reject_score=self.reject(ex_chosen)
#            print "classify score", classify_score
#            print "reject_score", reject_score

            w=(1.0-eeta_w*lambdaa_w)*w
            u=(1.0-eeta_u*lambdaa_u)*u
            
            l1=1.0+(a/2.0)*(reject_score-classify_score*ex_chosen.label)
            l2=c*(1.0-b*reject_score)

            if l1>max(l2,0):
                w+=eeta_w*ex_chosen.gamma*a*ex_chosen.label*ex_chosen.features_w/2
                u-=eeta_u*ex_chosen.gamma*a*ex_chosen.features_u/2
                
            elif l2>max(l1,0):  
                u+=eeta_u*ex_chosen.gamma*c*b*ex_chosen.features_u
                
            wbar=(1.0-rho)*wbar+rho * w #normalization
            ubar=(1.0-rho)*ubar+rho * u
            
            self.w=w
            self.u=u
           
            if (t)%len(examples)==0:
                loss[curr_epoch]=0
                for ex in examples:
                    rs=self.reject(ex)
                    cs=self.classify(ex)
                    l1=1.0+(a/2.0)*(rs-cs*ex.label)
                    l2=c*(1.0-b*rs)
                    loss[curr_epoch]+=max(0,l1,l2)

                curr_epoch+=1
                shuffle(examples)
                self.w=wbar
                self.u=ubar
        self.w=wbar
        self.u=ubar
        self.losses=loss
        
        
#############################################################33333##############################        
        
        
        
    def train3(self, examples,**kwargs): # without wbar
#        print "c=", self.c 
        
        siz1=np.shape(examples[0].features_w)[0]
        siz2=np.shape(examples[0].features_u)[0]
        
        w=np.array(np.zeros(siz1)) 
        wbar=np.array(np.zeros(siz1)) 
        
        u=np.array(np.zeros(siz2))
        ubar=np.array(np.zeros(siz2)) 
        
        self.w=wbar
        self.u=ubar

        epochs=self.epochs
        
        T=(len(examples))*epochs
        lambdaa_u=self.Lambda_u
        lambdaa_w=self.Lambda_w
       
        a=self.alpha
        b=self.beta
        c=self.c
        loss=np.zeros(epochs)
        curr_epoch=0
        for t in range(1,T+1):
            rho=2.0/(t+2.0)
            
            eeta_u=1.0/(lambdaa_u*t)
            eeta_w=1.0/(lambdaa_w*t)
            ex_chosen=examples[(t-1)%len(examples)]
#            import pdb; pdb.set_trace()
            classify_score=self.classify(ex_chosen)
            reject_score=self.reject(ex_chosen)
            
            w=(1.0-eeta_w*lambdaa_w)*w
            u=(1.0-eeta_u*lambdaa_u)*u
            
            l1=1.0+(a/2.0)*(reject_score-classify_score*ex_chosen.label)
            l2=c*(1.0-b*reject_score)

            if l1>max(l2,0):
                w+=eeta_w*ex_chosen.gamma*a*ex_chosen.label*ex_chosen.features_w/2             
                u-=eeta_u*ex_chosen.gamma*a*ex_chosen.features_u/2
         
            elif l2>max(l1,0):  
                u+=eeta_u*ex_chosen.gamma*c*b*ex_chosen.features_u
                
            wbar=(1.0-rho)*wbar+rho * w
            ubar=(1.0-rho)*ubar+rho * u
            
            self.w=w
            self.u=u
           
            if (t)%len(examples)==0:
                loss[curr_epoch]=0
                for ex in examples:
                    rs=self.reject(ex)
                    cs=self.classify(ex)
                    l1=1.0+(a/2.0)*(rs-cs*ex.label)
                    l2=c*(1.0-b*rs)
                    loss[curr_epoch]+=max(0,l1,l2)

                curr_epoch+=1
                shuffle(examples)

        self.losses=loss     
        
        
        
        
        
        
    def train2(self, examples,**kwargs): # choose equal number of positive and negative examples
        
        
        siz1=np.shape(examples[0].features_w)[0]
        siz2=np.shape(examples[0].features_u)[0]
        
        ex_p=[]
        ex_n=[]
        
        for e in examples:
            if e.label<0:
                ex_n+=[e]
            else:
                ex_p=[e]
        
        
        w=np.array(np.zeros(siz1)) 
        wbar=np.array(np.zeros(siz1)) 
        
        u=np.array(np.zeros(siz2))
        ubar=np.array(np.zeros(siz2)) 
        
        self.w=wbar
        self.u=ubar

        epochs=self.epochs
        
        T=(len(examples))*epochs
        lambdaa_u=self.Lambda_u
        lambdaa_w=self.Lambda_w

        
        a=self.alpha
        b=self.beta
        c=self.c
        loss=np.zeros(epochs)
        curr_epoch=0
        
        pos_ind=0
        neg_ind=0
        toss=None
        for t in range(1,T+1):
            rho=2.0/(t+2.0)
            toss=random.choice([1.0, -1.0])
            eeta_u=1.0/(lambdaa_u*t)
            eeta_w=1.0/(lambdaa_w*t)
            ex_chosen=None
            if toss>0:
                ex_chosen=ex_p[pos_ind%len(ex_p)]
                pos_ind+=1

            else:
                ex_chosen=ex_n[neg_ind%len(ex_n)]
                neg_ind+=1

            classify_score=self.classify(ex_chosen)
            reject_score=self.reject(ex_chosen)


            u=(1.0-eeta_u*lambdaa_u)*u
            w=(1.0-eeta_w*lambdaa_w)*w
            
            l1=1.0+(a/2.0)*(reject_score-classify_score*ex_chosen.label)
            l2=c*(1.0-b*reject_score)

            if l1>max(l2,0):
                w+=eeta_w*ex_chosen.gamma*a*ex_chosen.label*ex_chosen.features_w/2
                
                
                u-=eeta_u*ex_chosen.gamma*a*ex_chosen.features_u/2
                
                
                
            elif l2>max(l1,0):  
                u+=eeta_u*ex_chosen.gamma*c*b*ex_chosen.features_u
                
            
            wbar=(1.0-rho)*wbar+rho * w
            ubar=(1.0-rho)*ubar+rho * u
            self.w=w
            self.u=u
            

           
            if (t)%len(examples)==0:
                loss[curr_epoch]=0
                for ex in examples:
                    rs=self.reject(ex)
                    cs=self.classify(ex)
                    l1=1.0+(a/2.0)*(rs-cs*ex.label)
                    l2=c*(1.0-b*rs)
                    loss[curr_epoch]+=max(0,l1,l2)

                curr_epoch+=1
                
                
        self.w=wbar
        self.u=ubar
          

        self.losses=loss
                
#############################################################################################


