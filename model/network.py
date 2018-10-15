import numpy as np
import pickle 
import sys
from time import *
from model.loss import *
from model.layers import *

class Net:
    def __init__(self):
        # Lenet
        # input: 28x28
        # conv1: (5x5x6)@s1p2 -> 28x28x6 {(28-5+2x2)/1+1}
        # maxpool2: (2x2)@s2 -> 14x14x6 {(28-2)/2+1}
        # conv3: (5x5x16)@s1p0 -> 10x10x16 {(14-5)/1+1}
        # maxpool4: (2x2)@s2 -> 5x5x16 {(10-2)/2+1}
        # conv5: (5x5x120)@s1p0 -> 1x1x120 {(5-5)/1+1}
        # fc6: 120 -> 84
        # fc7: 84 -> 10
        # softmax: 10 -> 10
        lr = 0.01
        self.layers = []
        self.layers.append(Convolution2D(inputs_channel=1, num_filters=6, kernel_size=5, padding=2, stride=1, learning_rate=lr, name='conv1'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool2'))
        self.layers.append(Convolution2D(inputs_channel=6, num_filters=16, kernel_size=5, padding=0, stride=1, learning_rate=lr, name='conv3'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool4'))
        self.layers.append(Convolution2D(inputs_channel=16, num_filters=120, kernel_size=5, padding=0, stride=1, learning_rate=lr, name='conv5'))
        self.layers.append(ReLu())
        self.layers.append(Flatten())
        self.layers.append(FullyConnected(num_inputs=120, num_outputs=84, learning_rate=lr, name='fc6'))
        self.layers.append(ReLu())
        self.layers.append(FullyConnected(num_inputs=84, num_outputs=10, learning_rate=lr, name='fc7'))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)

    def train(self, training_data, training_label, batch_size, epoch, weights_file):
        total_acc = 0
        for e in range(epoch):
            for batch_index in range(0, training_data.shape[0], batch_size):
                # batch input
                if batch_index + batch_size < training_data.shape[0]:
                    data = training_data[batch_index:batch_index+batch_size]
                    label = training_label[batch_index:batch_index + batch_size]
                else:
                    data = training_data[batch_index:training_data.shape[0]]
                    label = training_label[batch_index:training_label.shape[0]]
                loss = 0
                acc = 0
                start_time = time()
                for b in range(batch_size):
                    x = data[b]
                    y = label[b]
                    # forward pass
                    for l in range(self.lay_num):
                        output = self.layers[l].forward(x)
                        x = output
                    loss += cross_entropy(output, y)
                    if np.argmax(output) == np.argmax(y):
                        acc += 1
                        total_acc += 1
                    # backward pass
                    dy = y
                    for l in range(self.lay_num-1, -1, -1):
                        dout = self.layers[l].backward(dy)
                        dy = dout
                # time
                end_time = time()
                batch_time = end_time-start_time
                remain_time = (training_data.shape[0]*epoch-batch_index-training_data.shape[0]*e)/batch_size*batch_time
                hrs = int(remain_time)/3600
                mins = int((remain_time/60-hrs*60))
                secs = int(remain_time-mins*60-hrs*3600)
                # result
                loss /= batch_size
                batch_acc = float(acc)/float(batch_size)
                training_acc = float(total_acc)/float((batch_index+batch_size)*(e+1))
                print('=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.2f} === BAcc: {4:.2f} === TAcc: {5:.2f} === Remain: {6:d} Hrs {7:d} Mins {8:d} Secs ==='.format(e,epoch,batch_index+batch_size,loss,batch_acc,training_acc,int(hrs),int(mins),int(secs)))
        # dump weights and bias
        obj = []
        for i in range(self.lay_num):
            cache = self.layers[i].extract()
            obj.append(cache)
        with open(weights_file, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def test(self, data, label, test_size):
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size)/float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size)/float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                #sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
                #sys.stdout.write("\b" * (toolbar_width-st+2))
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc)/float(test_size)))


    def test_with_pretrained_weights(self, data, label, test_size, weights_file):
        with open(weights_file, 'rb') as handle:
            b = pickle.load(handle)
        self.layers[0].feed(b[0]['conv1.weights'], b[0]['conv1.bias'])
        self.layers[3].feed(b[3]['conv3.weights'], b[3]['conv3.bias'])
        self.layers[6].feed(b[6]['conv5.weights'], b[6]['conv5.bias'])
        self.layers[9].feed(b[9]['fc6.weights'], b[9]['fc6.bias'])
        self.layers[11].feed(b[11]['fc7.weights'], b[11]['fc7.bias'])
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size)/float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size)/float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                #sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
                #sys.stdout.write("\b" * (toolbar_width-st+2))
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc)/float(test_size)))
        
    def predict_with_pretrained_weights(self, inputs, weights_file):
        with open(weights_file, 'rb') as handle:
            b = pickle.load(handle)
        self.layers[0].feed(b[0]['conv1.weights'], b[0]['conv1.bias'])
        self.layers[3].feed(b[3]['conv3.weights'], b[3]['conv3.bias'])
        self.layers[6].feed(b[6]['conv5.weights'], b[6]['conv5.bias'])
        self.layers[9].feed(b[9]['fc6.weights'], b[9]['fc6.bias'])
        self.layers[11].feed(b[11]['fc7.weights'], b[11]['fc7.bias'])
        for l in range(self.lay_num):
            output = self.layers[l].forward(inputs)
            inputs = output
        digit = np.argmax(output)
        probability = output[0, digit]
        return digit, probability


       
