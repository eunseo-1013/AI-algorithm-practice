import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn


X="i love you"
X_indices=[1,2,3] #X에 대한 단어 인덱스
dm=512
h=8
N=3
seq_len=len(X_indices)
dv=dm//h
dq=dm//h
dk=dm//h

wq_n=torch.randn(N,dm, dm,  requires_grad=True)
wk_n=torch.randn(N,dm, dm, requires_grad=True)
wv_n=torch.randn(N,dm, dm,requires_grad=True)
w0_n=torch.randn(N,dm, dm,requires_grad=True)
w1_n=torch.randn(N,dm, dm*4, requires_grad=True)
w2_n=torch.randn(N,dm*4, dm, requires_grad=True)


b1_n=torch.zeros(N,seq_len,dm*4, requires_grad=True)
b2_n=torch.zeros(N,seq_len,dm, requires_grad=True)

# 1. 단어 사전(Vocabulary) 크기 정의 (예: 1000개 단어)
vocab_size = 1000
# 2. 임베딩 층 만들기 (단어번호 하나를 넣으면 512차원 벡터가 나옴)
embedding_layer = nn.Embedding(vocab_size, dm)




def input_embedding(X_index): #입력값은 단어 인덱스
    input_x=embedding_layer(torch.LongTensor(X_index))
    return input_x


embedded_X=input_embedding(X_indices)


def softmax(x):
    e_x =torch.exp(x - torch.max(x, axis=-1, keepdims=True)[0])
    return e_x / e_x.sum(axis=-1, keepdims=True)

def PE(pos,i):
    if(i%2==0):
        return math.sin(pos/10000**(2*i/dm))
    else:
        return math.cos(pos/10000**(2*i/dm))


def Add_Norm(x,y):
    x=(x+y) #add
    for i in range(x.shape[0]):
        mean=torch.mean(x[i])
        std=torch.std(x[i])
        x[i]=(x[i]-mean)/(std+1e-6)
    return x

def Attention(qwi,kwi,vwi):
    score=qwi@kwi.T
    score=score/math.sqrt(dv)
    head=softmax(score)@vwi
    return head

def MHA(Q,K,V):
    heads=[]
    for i in range(h):
        qwi= Q@wq[:, dk*i:dk*(i+1)]
        kwi= K@wk[:, dk*i:dk*(i+1)]
        vwi= V@wv[:, dk*i:dk*(i+1)]
        heads.append(Attention(qwi,kwi,vwi))
    head=torch.concat(heads,axis=-1)
    return head@w0

def FFN(x):
    x_=torch.relu(x@w1)@w2
    return x_


#-----------<encoder>---------------



#1.embedding
embedded_X=input_embedding(X_indices)
Positional_encoding=torch.zeros(seq_len,dm)
for pos in range(seq_len):
    for i in range(dm):
        Positional_encoding[pos][i]=PE(pos,i)

PE_X=Positional_encoding + embedded_X

for n in range(N):
    wq=wq_n[n]
    wk=wk_n[n]
    wv=wv_n[n]
    w0=w0_n[n]
    w1=w1_n[n]
    w2=w2_n[n]
    b1=b1_n[n]
    b1=b1_n[n]
    #2.Q,K,V 초기값
    Q=PE_X@wq #seq_len,dq
    K=PE_X@wk
    V=PE_X@wv

    MHA_output=MHA(Q,K,V)
    Add_Norm_output=Add_Norm(PE_X,MHA_output)
    FFN_output=FFN(Add_Norm_output)
    Y=Add_Norm(Add_Norm_output,FFN_output)
    print(n, Y.shape)
    PE_X=Y




#-----------<decoder>---------------

