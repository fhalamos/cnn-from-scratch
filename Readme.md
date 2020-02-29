In this assignment you will practice writing backpropagation code, and training Convolutional Neural Networks. The goals of this assignment are as follows:

understand Neural Networks and how they are arranged in layered architectures
understand and be able to implement (vectorized) backpropagation
implement Convolutional Layer, Fully-Connected, MaxPool, Batch Normalization, ReLU, and CrossEntropy Loss With Softmax.
understand the architecture of Convolutional Neural Networks and get practice with training these models on data
gain experience with a major deep learning framework PyTorch.
To achieve these goals, this assignment mainly includes two sections:

(1) PyNet, a numpy-based CNN training framework (42 points).

(2) PyTorch on CIFAR-10 (30 points).

Notes on PyNet:

In this section,  you need to implement both forward and backward functions for many commonly used neural network layers.  For implementation, it is recommended to assign some intermediate values to class members (self.) so that you can reuse these variables for backward computation.  In some layers, you can write concise gradient formula by borrowing intermediate variables from the forward computation. We provide a self-checker function to compare your output (both forward and backward) with the PyTorch counterpart, so as to numerically validate your implementation; we'll grade your code mainly based on the differences with PyTorch output. Since neural networks aims to process large chunk of data, you are encouraged to vectorize your code to maximize running speed. 

Notes on Pytorch on CIFAR-10:

- Only Python3 and Pytorch version >=1.0 are supported. (see https://pytorch.org/ (Links to an external site.) for install instructions)

- In hw3_pytorch_cifar.py, you should complete:

(1) Training code (6 points); Testing code (6 points) including saving & loading models.

(2) Model design (12 points) & optimizer (3 points)

(3) Describe what you did, any additional features that you implemented, and/or any graphs that you made in the process of training and evaluating your network, final test accuracy @10 epochs in a writeup hw3.pdf (3 points).

- In model design, for training efficiency (since most of you will train models on CPU), a lightweight convnet is preferred (with fewer conv layer numbers,  channels, etc.). Your net can get full marks if test accuracy reaches at least 70% within 10 training epochs. Hint: you can start from building a network with only 3-4 conv layers and modify the architecture design if needed.

Resource Download:

hw3.zip

Submission:

You only need to submit

(1) hw3_pynet.py We release a self_checker script (like hw1) called gradient_checker.py for your debug usage. Please include the final checker results screenshot in writeup hw3.pdf. We will grade based on this screenshot and outputs of a different TA-version checker script.

(2) hw3_pytorch_cifar.py Complete PyTorch code based on the given code skeleton, conduct experiments on CIFAR-10 data.

(3) hw3.pdf.
- For (1), include a gradient_checker results screenshot.

- For (2), describe what you did, any additional features that you implemented, and/or any graphs that you made in the process of training and evaluating your network, final test accuracy @10 epochs.