# Handwritten Digit Recognition Using Convolutional Neural Network

<img src="img/MNIST.png" width="80%"/>

This repo builds a convolutional neural network based on [LENET](http://yann.lecun.com/exdb/lenet/) from scratch to recognize the MNIST Database of handwritten digits.

<img src="img/model.png"/>

## Getting Started

This example is only based on the python library ```numpy``` to implement convolutional layers, maxpooling layers and fully-connected layers, also including backpropagation and gradients descent to train the network and cross entropy to evaluate the loss.

## Running the Codes

```python main.py```

In the ```main.py```, you can modify the learning rate, epoch and batch size to train the CNN from scratch and evaluate the result. Besides, there is a provided pretrained weight file ```pretrained_weights.pkl```.

```python app.py```
This is the demo to predict handwritten digits based on the python api ```flask``` to build a localhost website.

<img src="img/demo.gif" width="70%"/>

## Results

* learning rate: 0.01
* batch size: 100
* training accuracy: 0.94
* loss
<img src="img/loss.png" width="70%"/>

## Blog Post
https://medium.com/deep-learning-g/build-lenet-from-scratch-7bd0c67a151e
