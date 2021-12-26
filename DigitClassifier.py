#!/usr/bin/env python3

from mnist import MNIST
import gzip
import matplotlib.pyplot as plt
import numpy as np

def load_data(train_in, train_lab, test_in, test_lab, image_size = 28, trainsize = 60000, testsize = 10000):
    
    #Reading Training Images
    f = gzip.open(train_in, 'r')
    f.read(16)
    buf = f.read(image_size * image_size * trainsize)
    data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32) 
    train_input = data.reshape(trainsize, image_size * image_size, 1)

    #Reading Testing Images
    f = gzip.open(test_in, 'r')
    f.read(16)
    buf = f.read(image_size * image_size * testsize)
    data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)
    test_input = data.reshape(testsize, image_size * image_size, 1)

    #Reading Training Labels
    f = gzip.open(train_lab, 'r')
    f.read(8)
    train_label = []
    for i in range(trainsize):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype = np.uint8).astype(np.int64)
        train_label.append(labels)

    #Reading Testing Labels
    f = gzip.open(test_lab, 'r')
    f.read(8)
    test_label = []
    for i in range(testsize):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype = np.uint8).astype(np.int64)
        test_label.append(labels)

    return train_input, train_label, test_input, test_label

def trainset_prep(train_in, train_lab, trainsize = 60000):
    X = train_in.T.squeeze()
    #print(np.shape(train_lab))
    Y = np.zeros((10, trainsize))
    for i in range(trainsize):
        #print(label[0])
        #print(train_lab[i][0])
        Y[train_lab[i][0]][i] = 1
        #print(np.array(y).reshape(1,10))
    #print(np.shape(Y))
    return X, Y

def make_mini_batch(X, Y, testsize = 60000, batchsize = 128):
    num_batch = testsize//batchsize

    Xr = np.delete(X, np.s_[0:testsize - (testsize%batchsize)], axis = 1)
    #print(np.shape(Xr))
    Yr = np.delete(Y, np.s_[0:testsize - (testsize%batchsize)], axis = 1)
    #print(np.shape(Yr))

    X = np.delete(X, np.s_[testsize - (testsize%batchsize):testsize], axis = 1)
    X = np.hsplit(X, num_batch)

    Y = np.delete(Y, np.s_[testsize - (testsize%batchsize):testsize], axis = 1)
    Y = np.hsplit(Y, num_batch)
    
    return num_batch, X, Y, Xr, Yr

def dReLU(Z):
    return Z > 0

def ReLU(Z):
    return np.maximum(Z, 0)

def SM(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0)

def dSM(Z):
    return (SM(Z)*(1-SM(Z)))

def Lf(A,Y):
    #print(np.shape(A), np.shape(Y))
    val = (-1*Y*np.log(A)) + ((1-Y)*np.log(1-A))
    #print(val)
    return val

#train_in, train_lab, test_in, test_lab = load_data(train_in = 'train-images-idx3-ubyte.gz', train_lab = 'train-labels-idx1-ubyte.gz', test_lab = 't10k-labels-idx1-ubyte.gz', test_in = 't10k-images-idx3-ubyte.gz')
#X, Y = trainset_prep(train_in, train_lab)
#print(np.shape(X))
#
#mean = np.sum(X, axis = 1).reshape(784,1) / 60000
#X = X - mean
#
#var = np.sum(X**2, axis = 1).reshape(784,1) / 60000
#for i in range(len(var)):
#    if var[i] != 0:
#        X[i] = X[i] / var[i]
#
#num_batch, X, Y, Xr, Yr = make_mini_batch(X,Y, testsize = 60000, batchsize = 128)

def set_hyperparameters():
    L = 4
    n = [784, 9, 5, 10]
    m = 128
    W = [[]]
    B = [[]]

    for l in range(1,L):
        Wl = np.random.rand(n[l], n[l-1]) * 0.01
        bl = np.zeros((n[l], 1), dtype = float)
        W.append(Wl)
        B.append(bl)

    return L, n, W, B, m

def reset_ZA(L, n, Xt, batchsize = 128):
    Z = []
    Z.append(Xt)
    A = []
    A.append(Xt)
    #print(Z[0], np.shape(Xt))
    for l in range(1,L):
        Zl = np.zeros((n[l], batchsize), dtype = float)
        Al = np.zeros((n[l], batchsize), dtype = float)
        Z.append(Zl)
        A.append(Al)
    return Z, A

def forward_propagation(L, n, batchsize, W, B, Z, A):
    for i in range(1,len(Z)):
        Z[i] = np.dot(W[i], A[i-1]).reshape(n[i], batchsize) + B[i]
        if (i == L-1):
            A[i] = SM(Z[i])
            #print(A[i])
        else:
            A[i] = ReLU(Z[i])
    return Z, A

def cost_calc(Y_hat, Y):
    batchsize = Y_hat.shape[1]
    cost = (np.dot(Y, np.log(Y_hat).T) + np.dot(1-Y, np.log(1-Y_hat).T)) * (-1/batchsize)
    return np.squeeze(cost)

def get_accuracy(Y_hat, Y):
    prob = np.copy(Y_hat)
    prob[prob > 0.5] = 1
    prob[prob <= 0.5] = 0
    #print(prob.shape, Y.shape, prob[:,0])
    return (prob == Y).all(axis = 0).mean()

def backward_propagation(A, Z, Y, W, B, L, n):
    batchsize = Y.shape[1]
    #print(Y.shape, A[L-1].shape)
    dA = []
    da = -(np.divide(Y, A[L-1]) - np.divide(1-Y, 1-A[L-1]))
    dA.append(da)
    dZ = [[]]
    dW = [[]]
    dB = [[]]
    
    #Calculating dZ for all layers
    for i in range(L-1, 0, -1):
        if (i == L-1):
            dz = A[i] - Y
        else:
            dz = np.dot(W[i+1].T,dZ[0]) * dReLU(Z[i])
        dZ.insert(0, dz)
        #print(dZ[0].shape)

    #Caluculating dW, dB for all layers
    for i in range(1,L):
        dw = np.dot(dZ[i-1], A[i-1].T) / batchsize
        db = np.sum(dZ[i-1], axis = 1, keepdims = True) / batchsize
        #print(i, dw.shape, db.shape)
        
        dW.append(dw)
        dB.append(db)

    return dW, dB

#dZ3 = A[3] - Y
#print(dZ3.shape)
#dW3 = np.dot(dZ3, A[2].T) / batchsize
#print(dW3.shape, W[3].shape)
#dB3 = np.sum(dZ3, axis = 1, keepdims = True) / batchsize

#dZ2 = np.dot(W[3].T, dZ3) * dReLU(Z[2])
#print(dZ2.shape)
#dW2 = np.dot(dZ2, A[1].T) / batchsize
#dB2 = np.sum(dZ2, axis = 1, keepdims = True) / batchsize

#dZ1 = np.dot(W[2].T, dZ2) * dReLU(Z[1])
#print(dZ1.shape)
#dW1 = np.dot(dZ1, A[0].T) / batchsize
#dB1 = np.sum(dZ1, axis = 1, keepdims = True) / batchsize
        
def basicNN():
    #Mini batchs
    #Forward Propagation
    #Cost Calculation
    #Backward Propagation

    L, n, W, B, batchsize = set_hyperparameters()

    #Loading Parameters
    train_in, train_lab, test_in, test_lab = load_data(train_in = 'train-images-idx3-ubyte.gz', train_lab = 'train-labels-idx1-ubyte.gz', test_lab = 't10k-labels-idx1-ubyte.gz', test_in = 't10k-images-idx3-ubyte.gz')
    X, Y = trainset_prep(train_in, train_lab)
    _, Batch = (np.shape(X))
    mean = np.sum(X, axis = 1).reshape(784,1) / Batch
    #print(np.shape(mean))
    #X = X - mean
    #
    #var = np.sum(X**2, axis = 1).reshape(784,1) / Batch
    ##print(np.shape(var))
    #for i in range(len(var)):
    #    if var[i] != 0:
    #        X[i] = X[i] / var[i]

    num_batch, X, Y, Xr, Yr = make_mini_batch(X,Y, testsize = Batch, batchsize = 128)
    cost_history = []
    accuracy_history = []

    nepoch = 100
    for epoch in range(nepoch):
        for i in range(num_batch):
            Xt = X[i]
            Yt = Y[i]

            Z, A = reset_ZA(L, n, Xt, batchsize)
            #Forward Propagation
            Z, A = forward_propagation(L, n, batchsize, W, B, Z, A)

            #Cost Calculation
            J = cost_calc(Y_hat = A[L-1],Y = Yt)
            cost_history.append(J)
            Accuracy = get_accuracy(Y_hat = A[L-1], Y = Yt)
            accuracy_history.append(Accuracy)

            #Backward Propagation
            dW, dB = backward_propagation(A, Z, Yt, W, B, L, n)

            #Update Parameters
            α = 0.01
            for i in range(1, L):
                W[i] = W[i] - α*dW[i]
                B[i] = B[i] - α*dB[i]
    return W, B, cost_history, accuracy_history, test_in, test_lab

W, B, cost, acc, test_in, test_lab = basicNN()

def testing(L, n, W, B, test_in, test_lab, batchsize = 10000):
    test_in = test_in.T.squeeze()
    Y = np.zeros((10, batchsize))
    for i in range(batchsize):
        Y[test_lab[i][0]][i] = 1
    cost_history = []
    accuracy_history = []
    Z, A = reset_ZA(L, n, test_in, batchsize)
    Z, A = forward_propagation(L, n, batchsize, W, B, Z, A)
    J = cost_calc(Y_hat = A[L-1],Y = Y)
    cost_history.append(J)
    Accuracy = get_accuracy(Y_hat = A[L-1], Y = Y)
    accuracy_history.append(Accuracy)

    print("Test Set Error : ", np.sum(accuracy_history)/len(accuracy_history))
    #print(np.sum(J)/len(J))

L, n, _, _, _ = set_hyperparameters()
testing(L,n, W, B, test_in, test_lab, batchsize = 10000)
#plt.plot([i for i in range(len(acc))],acc)
#plt.plot([i for i in range(len(cost))], cost)
print("Train Set Error : ", round(np.sum(acc[-1280:])/1280,4))
#plt.show()
#print(acc)
print("Length of acc : ", len(acc))
A = [0]
beta = 0.9921875
for i in range(len(acc)):
    A.append(beta*A[i] + (1-beta)*acc[i])
plt.plot([i for i in range(len(A))], A)
plt.show()
