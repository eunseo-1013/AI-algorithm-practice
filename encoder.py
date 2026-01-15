import numpy as np
import pandas as pd
import math
import torch

dm=512
h=8
N=1
dv,dq,dk=dm/h
wq,wk,wv=np.zeros(dm,dm//h)
w0=np.zeros(dm,dm)
w1=np.zeros(dm,dm*4)
w2=np.zeros(dm*4,dm)
b1,b2=0

X="i love you"


def input_embedding(X):
    input_x=0 #임베딩어케함
    return input_x

Q=input_embedding(X)@wq
K=input_embedding(X)@wk
V=input_embedding(X)@wv

def PE(pos,i):
    if(i&2==0):
        return math.sin(pos/10000**(2*i/dm))
    else:
        return math.cos(pos/10000**(2*i/dm))


def Add_Norm(x,y):
    x=x+y #add
    for i in range(x.shape[0]):
        x[i]=x[i].Standard()? #norm
    return x

def Attention(qwi,kwi,vwi):
    score=qwi*kwi
    score=score/math.sqr(dv)
    head=softmax(score)@vwi
    return head

def MHA(Q,K,V):
    v_=0
    for i in range(h):
        qwi= Q@wq[dq*i]
        kwi= K@wk[dq*i]
        vwi= V@wv[dq*i]
        v_= np.concat(Attention(qwi,kwi,vwi))
    return v_@w0

def FFN(x):
    x_=ReLu(x@w1+b1)@w2+b2
    return x_