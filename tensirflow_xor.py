# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:23:23 2021

@author: jwkim
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
tf.random.set_seed(678)

x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y=np.array([0.,1.,1.,0.])
model=Sequential()
model.add(Dense(units=2,activation='sigmoid',input_dim=2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
print(model.summary())
model.fit(x,y,epochs=50000,batch_size=4,verbose=0)
print(model.predict(x,batch_size=4))

import math
def sigmoid(x):
    return 1/(1+math.exp(-x))
def get_output(x):
    layer0=model.layers[0]
    layer0_weights,layer0_bias=layer0.get_weights()
    layer0_node0_weights=np.transpose(layer0_weights)[0]
    layer0_node0_bias=layer0_bias[0]
    layer0_node0_output=sigmoid(np.dot(x,layer0_node0_weights)+layer0_node0_bias)
    
    layer0_node1_weights=np.transpose(layer0_weights)[1]
    layer0_node1_bias=layer0_bias[1]
    layer0_node1_output=sigmoid(np.dot(x,layer0_node1_weights)+layer0_node1_bias)
    
    layer1=model.layers[1]
    layer1_weights,layer1_bias=layer1.get_weights()
    layer1_output=sigmoid(np.dot([layer0_node0_output,layer0_node1_output],layer1_weights)+layer1_bias)
    
    print(layer1_output)
get_output([0,0])
get_output([0,1])
get_output([1,0])
get_output([1,1])