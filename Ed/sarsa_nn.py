# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:04:16 2018

@author: ebarker
"""

# ipython only (?)
%reset 

##########################################
#  Code to store and update CNN weights  #
##########################################

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os

## GENERAL FUNCTIONS

def intToBool(num, digits):
    return np.asarray([bool(num & (1<<n)) for n in range(digits)],dtype=np.int32)

def boolToInt(array):
    x = 0
    for i in range(len(array)):
        x = array[i] << i | x
    return x

def epsilonGreedy(q, epsilon):
    if (random.random() < epsilon):
        return random.randint(1,len(q))
    else:
        return np.argmax(q) + 1


## TEMP ENVIRONMENT (usually provided externally)

# global parameters

ZX = 8
ZY = 1
T = 1000000
intT = T/100
beta = 0.2

# environment

S = 2**ZX
A = 2**ZY
zeta = 6 # for Bernoilli style reward function
#R = np.random.normal(0,1,[S,A])
R = np.random.binomial(1,zeta/S,[S,A]) * 250
P = np.random.randint(1,S,[S,A])

Gamma = np.empty((0,ZX), int)

for i in range(S):
    Gamma = np.append(Gamma,np.reshape(intToBool(i,ZX),[1,ZX]),axis=0)

# environment update

def genInput(y, old_s):
    a = boolToInt(y) + 1
    new_s = P[old_s - 1, a - 1]
    x = intToBool(new_s - 1,ZX)
    r = R[old_s - 1, a - 1]
    return new_s, x, r


## AGENT PARAMETERS

# arguments provided externally

eta = 0.001
epsilon = 0.05
gamma = 0.99
nW1 = 30
nW2 = 100
nW3 = 100
nW4 = 20
stddev = 0.01
dropout = 0.5 # proportion to keep

print("\nTotal weights: ", ZX * nW1 + nW1 * nW2 + nW2 * nW3 + nW3 * nW4 + nW4 * A,"\n")

# need sharedObjects instance

# basic architecture

X = tf.placeholder('float', [1, ZX], name = 'X') # for input vector
D = tf.placeholder('float', [], name = 'D') # for temporal difference delta
OA = tf.placeholder('int32', [], name = 'OA') # for the action selected (to compute derivative)

W1 = tf.Variable(tf.random_normal([ZX, nW1], stddev=stddev), name = 'W1')
W1do = tf.nn.dropout(W1,dropout)
L1 = tf.nn.sigmoid(tf.matmul(X, W1do), name = 'L1')

W2 = tf.Variable(tf.random_normal([nW1, nW2], stddev=stddev), name = 'W2')
W2do = tf.nn.dropout(W2,dropout)
L2 = tf.nn.sigmoid(tf.matmul(L1, W2do), name = 'L2')

W3 = tf.Variable(tf.random_normal([nW2, nW3], stddev=stddev), name = 'W3')
W3do = tf.nn.dropout(W3,dropout)
L3 = tf.nn.sigmoid(tf.matmul(L2, W3do), name = 'L3')

W4 = tf.Variable(tf.random_normal([nW3, nW4], stddev=stddev), name = 'W4')
W4do = tf.nn.dropout(W4,dropout)
L4 = tf.nn.sigmoid(tf.matmul(L3, W4do), name = 'L4')

W5 = tf.Variable(tf.random_normal([nW4, A], stddev=stddev), name = 'W5')

Q = tf.matmul(L4, W5, name = 'Q')

nambla1 = tf.gradients(tf.slice(tf.reshape(Q,[A]), [OA], [1]), W1)[0]
nambla2 = tf.gradients(tf.slice(tf.reshape(Q,[A]), [OA], [1]), W2)[0]
nambla3 = tf.gradients(tf.slice(tf.reshape(Q,[A]), [OA], [1]), W3)[0]
nambla4 = tf.gradients(tf.slice(tf.reshape(Q,[A]), [OA], [1]), W4)[0]
nambla5 = tf.gradients(tf.slice(tf.reshape(Q,[A]), [OA], [1]), W5)[0]

delW1 = tf.multiply(D, nambla1)
delW2 = tf.multiply(D, nambla2)
delW3 = tf.multiply(D, nambla3)
delW4 = tf.multiply(D, nambla4)
delW5 = tf.multiply(D, nambla5)

incrW1 = tf.assign(W1, tf.add(W1, delW1))
incrW2 = tf.assign(W2, tf.add(W2, delW2))
incrW3 = tf.assign(W3, tf.add(W3, delW3))
incrW4 = tf.assign(W4, tf.add(W4, delW4))
incrW5 = tf.assign(W5, tf.add(W5, delW5))

# initialise environment

s = np.random.randint(1,S)
x = intToBool(s - 1,ZX)
r = 0

# session start

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

session.run(tf.global_variables_initializer())

# for tensorboard

train_writer = tf.summary.FileWriter(os.getcwd() + '/temp_tf_output', session.graph)
#tf.summary.scalar('Q', Q)
#tf.summary.scalar('D', D)

#################

# type into anaconda console >>tensorboard --logdir=C:/Users/ebarker/temp_tf_output --host=127.0.0.1
# then go here in Chrome >>http://localhost:6006

#################

## RUN ITERATIONS

# initialise other variables

q = session.run(Q, feed_dict={X: np.reshape(x,[1,ZX])}).flatten()
a = epsilonGreedy(q.flatten(), epsilon)
y = intToBool(a - 1, ZY)
    
# for plotting

xPlot = range(math.floor(T/intT))
rPlot = [0 for i in range(math.floor(T/intT))]
wPlot1 = [[[0 for i in range(math.floor(T/intT))] for j in range(ZX)] for k in range(nW1)]
wPlot2 = [[[0 for i in range(math.floor(T/intT))] for j in range(nW1)] for k in range(nW2)]
wPlot3 = [[[0 for i in range(math.floor(T/intT))] for j in range(nW2)] for k in range(nW3)]
wPlot4 = [[[0 for i in range(math.floor(T/intT))] for j in range(nW3)] for k in range(nW4)]
wPlot5 = [[[0 for i in range(math.floor(T/intT))] for j in range(nW4)] for k in range(A)]
counter = 0
rAvg = 0

fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()
fig3 = plt.figure()
ax3 = plt.axes()
fig4 = plt.figure()
ax4 = plt.axes()
fig5 = plt.figure()
ax5 = plt.axes()
fig6 = plt.figure()
ax6 = plt.axes()

for t in range(T):
    
    # save old variables
    
    q_old = q
    a_old = a
    x_old = x

    # generate update

    s, x, r = genInput(y, s)
    q = session.run(Q, feed_dict={X: np.reshape(x,[1,ZX])}).flatten()
    a = epsilonGreedy(q, epsilon)
    y = intToBool(a - 1, ZY)
    
    # create tensor from inputs
    
    delta = eta * (r + gamma * q[a - 1] - q_old[a_old - 1])
    
    # update weights
    
    session.run([incrW1, incrW2, incrW3, incrW4, incrW5], feed_dict={X: np.reshape(x_old,[1,ZX]), OA: a_old - 1, D: delta})
    
    # tracking for debugging
    
    #print(session.run(Q, feed_dict={X: np.reshape(x,[1,ZX]), D: delta}))
    #print(session.run(nambla1, feed_dict={X: np.reshape(x,[1,ZX]), OA: a_old - 1, D: delta}))
    
    rAvg = (rAvg * counter + r) / (counter + 1)
    
    counter = counter + 1
    
    if t%intT==0: 
        rPlot[math.floor(t/intT)] = rAvg
        tempW1 = W1.eval(session=session)
        tempW2 = W2.eval(session=session)
        tempW3 = W3.eval(session=session)
        tempW4 = W4.eval(session=session)
        tempW5 = W5.eval(session=session)
        for j in range(nW1):
            for k in range(ZX):
                wPlot1[j][k][math.floor(t/intT)] = tempW1[k,j]
        for j in range(nW2):
            for k in range(nW1):
                wPlot2[j][k][math.floor(t/intT)] = tempW2[k,j]
        for j in range(nW3):
            for k in range(nW2):
                wPlot3[j][k][math.floor(t/intT)] = tempW3[k,j]
        for j in range(nW4):
            for k in range(nW3):
                wPlot4[j][k][math.floor(t/intT)] = tempW4[k,j]
        for j in range(A):
            for k in range(nW4):
                wPlot5[j][k][math.floor(t/intT)] = tempW5[k,j]
        counter = 0
        print("iteration: ", t, " reward:", round(rAvg,5), " state:", s, " action:", a, " temporal difference: ", delta)
    

session.close()

ax1.plot(xPlot,rPlot)
for j in range(nW1):
    for k in range(ZX):
        ax2.plot(xPlot,wPlot1[j][k])
for j in range(nW2):
    for k in range(nW1):
        ax3.plot(xPlot,wPlot2[j][k])
for j in range(nW3):
    for k in range(nW2):
        ax4.plot(xPlot,wPlot3[j][k])
for j in range(nW4):
    for k in range(nW3):
        ax5.plot(xPlot,wPlot4[j][k])
for j in range(A):
    for k in range(nW4):
        ax6.plot(xPlot,wPlot5[j][k])
    
print("\nMean performance: ",np.mean(rPlot[math.floor(len(rPlot) * (1-beta)):len(rPlot)]), " ... compare sum(R):", sum(sum(R)))

