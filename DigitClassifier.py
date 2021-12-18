from mnist import MNIST
import gzip
import matplotlib.pyplot as plt
import numpy as np

f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 600

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

image = np.asarray(data[594]).squeeze()
plt.imshow(image)
#print(data[594])
#plt.show()



f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
label = []
for i in range(60000):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    label.append(labels)
    #print(labels)

print(label[594][0])

#for x in data[594]:
#    print()
#    for y in x:
#        print(y[0], end = '\t')


def basicNN(inputdata,m):
    #Forward Propagation, Backward Propagation, No-Minibatches, Adam Optimization, Learning Rate Decay
    L = 3
    n = [784,9,5,9]
    print(n)
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

    for _ in range(10):
        for i in range(L):
            if i == 0:
                mean = np.sum(X, axis = 1).reshape(n[i+1],1) / m
                X = X - mean
                var = np.sum(mat**2,axis = 1).reshape(n[i+1],i) / m
                X = X / var
                Z[i] = np.dot(W[i],X) + B[i]
            else:
                Z[i] = np.dot(W[i],A[i-1]) + B[i]
            if i != L-1:
                A[i] = AF(Z[i])
            else:
                A[i] = SM(Z[i])

#basicNN(data)

#9 784 784 10000
#784 9 9 100