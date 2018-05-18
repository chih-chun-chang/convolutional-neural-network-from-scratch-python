import numpy as np 

class Convolution2D:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F) 
        self.F = num_filters
        self.K = kernel_size
        self.C = inputs_channel

        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0,self.F):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)), size=(self.C, self.K, self.K))

        self.p = padding
        self.s = stride
        self.lr = learning_rate
        self.name = name

    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * size + w
        new_h = 2 * size + h
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out

    def forward(self, inputs):
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        C = inputs.shape[0]
        W = inputs.shape[1]+2*self.p
        H = inputs.shape[2]+2*self.p
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.p)
        WW = (W - self.K)/self.s + 1
        HH = (H - self.K)/self.s + 1
        feature_maps = np.zeros((self.F, WW, HH))
        for f in range(self.F):
            for w in range(WW):
                for h in range(HH):
                    feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.K,h:h+self.K]*self.weights[f,:,:,:])+self.bias[f]

        return feature_maps

    def backward(self, dy):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.K,h:h+self.K]
                    dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Maxpooling2D:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = (W - self.pool)/self.s + 1
        new_height = (H - self.pool)/self.s + 1
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(W/self.s):
                for h in range(H/self.s):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, w/self.pool, h/self.pool]
        return dx

    def extract(self):
        return 
    
class FullyConnected:

    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.lr = learning_rate
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T

    def backward(self, dy):

        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)

        self.weights -= self.lr * dw.T
        self.bias -= self.lr * db

        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Flatten:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        return inputs.reshape(1, self.C*self.W*self.H)
    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)
    def extract(self):
        return

class ReLu:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    def extract(self):
        return

class Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, dy):
        return self.out.T - dy.reshape(dy.shape[0],1)
    def extract(self):
        return

