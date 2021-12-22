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

#train_in, train_lab, test_in, test_lab = load_data(train_in = 'train-images-idx3-ubyte.gz', train_lab = 'train-labels-idx1-ubyte.gz', test_lab = 't10k-labels-idx1-ubyte.gz', test_in = 't10k-images-idx3-ubyte.gz')
#X, Y = trainset_prep(train_in, train_lab)
#
#print(np.shape(X))
#
#mean = np.sum(X, axis = 1).reshape(784,1) / 60000
#print(np.shape(mean))
#X = X - mean
#
#var = np.sum(X**2, axis = 1).reshape(784,1) / 60000
#print(np.shape(var))
#for i in range(len(var)):
#    if var[i] != 0:
#        X[i] = X[i] / var[i]
#
#num_batch, X, Y, Xr, Yr = make_mini_batch(X,Y, testsize = 60000, batchsize = 128)
#
#print(np.shape(X), num_batch)
#
#print(np.shape(Y), num_batch)

def basicNN():
    #Mini batchs
    #Forward Propagation
    #Cost Calculation
    #Backward Propagation
    #Adam Optimization
    #Learning Rate Decay
    
    L = 3 #No of Layers
    n = [784,9,5,10] #No of Nodes in each layer
    print(n)
    m = 128
    W = []
    b = []
    Z = []
    for i in range(1, L+1):
        Wi = np.random.rand(n[i],n[i-1]) * 0.01
        bi = np.zeros((n[i],1), dtype = float)
        Zi = np.zeros((n[i],m), dtype = float)
        Ai = np.zeros((n[i],m), dtype = float)
        W.append(Wi)
        B.append(bi)
        Z.append(Zi)
        A.append(Ai)
    print(W[1])
    print(np.shape(W[1]))


    train_in, train_lab, test_in, test_lab = load_data(train_in = 'train-images-idx3-ubyte.gz', train_lab = 'train-labels-idx1-ubyte.gz', test_lab = 't10k-labels-idx1-ubyte.gz', test_in = 't10k-images-idx3-ubyte.gz')
    X, Y = trainset_prep(train_in, train_lab)

    mean = np.sum(X, axis = 1).reshape(784,1) / 60000
    print(np.shape(mean))
    X = X - mean
    
    var = np.sum(X**2, axis = 1).reshape(784,1) / 60000
    print(np.shape(var))
    for i in range(len(var)):
        if var[i] != 0:
            X[i] = X[i] / var[i]

    batchsize = 128
    num_batch, X, Y, Xr, Yr = make_mini_batch(X,Y, testsize = 60000, batchsize = 128)

    nepoch = 100
    for epoch in range(nepoch):
        for i in range(num_batch):
            Xt = X[i]
            Yt = Y[i]
            #Forward Propagation
            Z1 = np.dot(W[0], Xt).reshape(n[1], batchsize) + B[0]
            A1 = ReLU(Z1)
            Z2 = np.dot(W[1],A1).reshape(n[2], batchsize) + B[1]
            A2 = ReLU(Z2)
            Z3 = np.dot(W[2],A2).reshape(n[3], batchsize) + B[2]
            A3 = SM(Z3)

            #Cost Calculation
            J = 0
            for i in range(batchsize):
                J += L(A3[i], Yt[i])
            J = J / batchsize
            #J = J + (lambd / (2*batchsize))*Feb(W)

            #Backward Propagation
            dZ3 = A3 - Yt[i]
            dW3 = np.dot(dZ3, A2.T) / batchsize
            db3 = np.sum(dZ3, axis = 1, keepdims = True) / batchsize
            dZ2 = np.dot(np.dot(W2.T,dZ3)) * dReLU(Z2)
            dW2 = np.dot(dZ2, A1.T) / batchsize
            db2 = np.sum(dZ2, axis = 1, keepdims = True) / batchsize
            dZ1 = np.dot(np.dot(W2.T,dZ3)) * dReLU(Z2)
            dW1 = np.dot(dZ1, X.T) / batchsize
            db1 = np.sum(dZ1, axis = 1, keepdims = True) / batchsize

            #Update Parameters
            α = 0.01
            W1 = W1 - α*dW1
            b1 = b1 - α*db1
            W2 = W2 - α*dW2
            b2 = b2 - α*db2
            W3 = W3 - α*dW3
            b3 = b3 - α*db3
    return W1,b1,W2,b2,W3,b3,mean,var