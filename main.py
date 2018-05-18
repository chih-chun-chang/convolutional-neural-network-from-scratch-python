import numpy as np
import mnist
from model.network import Net

print('Loadind data......')
num_classes = 10
train_images = mnist.train_images() #[60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print('Preparing data......')
train_images -= int(np.mean(train_images))
train_images /= int(np.std(train_images))
test_images -= int(np.mean(test_images))
test_images /= int(np.std(test_images))
training_data = train_images.reshape(60000, 1, 28, 28)
training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]

net = Net()
print('Training Lenet......')
net.train(training_data, training_labels, 32, 1, 'weights.pkl')
print('Testing Lenet......')
net.test(testing_data, testing_labels, 100)
print('Testing with pretrained weights......')
net.test_with_pretrained_weights(testing_data, testing_labels, 100, 'pretrained_weights.pkl')