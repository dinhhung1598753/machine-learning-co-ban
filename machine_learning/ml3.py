import pandas as pd
import numpy as np

data=pd.read_csv('data_classification.csv', header=None)
x= data.values[:,0:2]
y=data.values[:,2]
print(x)

def sigmoid(z):
    return 1.0 /(1+np.exp(-z))

def phan_chia(p):
    if p>= 0.5:
        return 1
    else:
        return 0

def predict(features, weights):
    z= np.dot(weights, features)
    return sigmoid(z)

def cost_function(features,labels, weights ):
   

    n = len(labels)

    predictions=predict(features,weights)

    cost_class1 = - labels*np.log(predictions)
    cost_class0 = -(1 - labels)*np.log(predictions)
    cost = cost_class0+ cost_class1
    return cost.sum()/n

def update_weights(features, labels, weights, learning_rate):
    n= len(labels)
    predictions = predict(features, weights)
    gd=np.dot(features.T-(predictions-labels))
    gd=gd/n
    gd= gd*learning_rate
    weights= weights-gd
    return weights

def train(features, labels, weights, learning_rate, iter):
    cost_his=[]
    for i in range(iter):
        weights = update_weights(features,labels,weights,learning_rate)
        cost=cost_function(features,labels,weights)
        cost_his.append(cost)
    return weights,cost_his
weights, cost_his=train(x,y,[0.03,0.03,0.03],0.001,50)
print(weights)
print(cost_his)


