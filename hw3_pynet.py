import numpy as np
from math import *

'''
    Linear

    Implementation of the linear layer (also called fully connected layer),
    which performs linear transformation on input data: y = xW + b.

    This layer has two learnable parameters:
        weight of shape (input_channel, output_channel)
        bias   of shape (output_channel)
    which are specified and initalized in the init_param() function.

    In this assignment, you need to implement both forward and backward
    computation.

    Arguments:
        input_channel  -- integer, number of input channels
        output_channel -- integer, number of output channels
'''
class Linear(object):

    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.input_channel,self.output_channel) * sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
        self.bias = np.zeros((self.output_channel))

    '''
        Forward computation of linear layer. (3 points)

        Note:  You may want to save some intermediate variables to class
        membership (self.) for reuse in backward computation.

        Arguments:
            input  -- numpy array of shape (N, input_channel)

        Output:
            output -- numpy array of shape (N, output_channel)
    '''
    def forward(self, input):

        self.X = input

        output = np.matmul(input, self.weight) + self.bias

        return output

    '''
        Backward computation of linear layer. (3 points)

        You need to compute the gradient w.r.t input, weight, and bias.
        You need to reuse variables from forward computation to compute the
        backward gradient.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel)

        Output:
            grad_input  -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_weight -- numpy array of shape (input_channel, output_channel), gradient w.r.t weight
            grad_bias   -- numpy array of shape (output_channel), gradient w.r.t bias

        Reference: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    '''
    def backward(self, grad_output):

        #loss_wrt_x = loss_wrt_y . y_wrt_X = grad_output . W^T
        grad_input = np.matmul(grad_output, self.weight.T)

        #loss_wrt_w = loss_wrt_y . y_wrt_w = X^T . grad_output
        grad_weight = np.matmul(self.X.T, grad_output)

        #loss_wrt_b = loss_wrt_y . y_wrt_b = 
        #mirar apuntes en cuaderno, pa estar seguro
        N, _ = grad_output.shape
        grad_bias = np.matmul(grad_output.T, np.ones(N))

        return grad_input, grad_weight, grad_bias

'''
    BatchNorm1d

    Implementation of batch normalization (or BN) layer, which performs
    normalization and rescaling on input data.  Specifically, for input data X
    of shape (N,input_channel), BN layers first normalized the data along batch
    dimension by the mean E(x), variance Var(X) that are computed within batch
    data and both have shape of (input_channel).  Then BN re-scales the
    normalized data with learnable parameters beta and gamma, both having shape
    of (input_channel).
    So the forward formula is written as:

        Y = ((X - mean(X)) /  sqrt(Var(x) + eps)) * gamma + beta

    At the same time, BN layer maintains a running_mean and running_variance
    that are updated (with momentum) during forward iteration and would replace
    batch-wise E(x) and Var(x) for testing. The equations are:

        running_mean = (1 - momentum) * E(x)   +  momentum * running_mean
        running_var =  (1 - momentum) * Var(x) +  momentum * running_var

    During test time, since the batch size could be arbitrary, the statistics
    for a batch may not be a good approximation of the data distribution.
    Thus, we instead use running_mean and running_var to perform normalization.
    The forward formular is modified to:

        Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta

    Overall, BN maintains 4 learnable parameters with shape of (input_channel),
    running_mean, running_var, beta, and gamma.  In this assignment, you need
    to complete the forward and backward computation and handle the cases for
    both training and testing.

    Arguments:
        input_channel -- integer, number of input channel
        momentum      -- float,   the momentum value used for the running_mean and running_var computation
'''
class BatchNorm1d(object):

    def __init__(self, input_channel, momentum = 0.9):
        self.input_channel = input_channel
        self.momentum = momentum
        self.eps = 1e-3
        self.init_param()

    def init_param(self):
        self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
        self.r_var = np.ones((self.input_channel)).astype(np.float32)
        self.beta = np.zeros((self.input_channel)).astype(np.float32)
        self.gamma = (np.random.rand(self.input_channel) * sqrt(2.0/(self.input_channel))).astype(np.float32)

    '''
        Forward computation of batch normalization layer and update of running
        mean and running variance. (3 points)

        You may want to save some intermediate variables to class membership
        (self.) and you should take care of different behaviors during training
        and testing.

        Arguments:
            input -- numpy array (N, input_channel)
            train -- bool, boolean indicator to specify the running mode, True for training and False for testing
    '''
    def forward(self, input, train): #isnt there a porblem for a variable to be called input?

        if(train):

            input_mean = input.mean(axis=0)
            input_var = input.var(axis=0)
            self.xhat = (input - input_mean)/np.sqrt(input_var+self.eps)
            output = self.xhat*self.gamma + self.beta

            self.r_mean = (1 - self.momentum) * input_mean + self.momentum * self.r_mean
            self.r_var =  (1 - self.momentum) * input_var + self.momentum * self.r_var

        else: #test
            output = (input - self.r_mean)/np.sqrt(self.r_var+self.eps)*self.gamma + self.beta

        return output

    '''
        Backward computation of batch normalization layer. (3 points)
        You need to compute gradient w.r.t input data, gamma, and beta.

        It is recommend to follow the chain rule to first compute the gradient
        w.r.t to intermediate variables, in order to simplify the computation.

        Arguments:
            grad_output -- numpy array of shape (N, input_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_gamma -- numpy array of shape (input_channel), gradient w.r.t gamma
            grad_beta  -- numpy array of shape (input_channel), gradient w.r.t beta
    
        Reference: https://kevinzakka.github.io/2016/09/14/batch_normalization/
    '''
    def backward(self, grad_output):
        
        N, _ = grad_output.shape
        
        grad_xhat = grad_output * self.gamma

        grad_input =    (1./N) * \
                        (1. / (np.sqrt(self.r_var + self.eps))) * \
                        (N * grad_xhat - np.sum(grad_xhat, axis=0) - self.xhat *np.sum(grad_xhat*self.xhat, axis=0))

        grad_beta = np.sum(grad_output, axis=0)
        grad_gamma = np.sum(self.xhat*grad_output, axis=0)

        return grad_input, grad_gamma, grad_beta

'''
    ReLU

    Implementation of ReLU (rectified linear unit) layer.  ReLU is the
    non-linear activation function that sets all negative values to zero.
    The formua is: y = max(x,0).

    This layer has no learnable parameters and you need to implement both
    forward and backward computation.

    Arguments:
        None
'''
class ReLU(object):
    def __init__(self):
        pass

    '''
        Forward computation of ReLU. (3 points)

        You may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input  -- numpy array of arbitrary shape

        Output:
            output -- numpy array having the same shape as input.
    '''
    def forward(self, input):

        output = np.maximum(0,input)
        return output

    '''
        Backward computation of ReLU. (3 points)

        You can either modify grad_output in-place or create a copy.

        Arguments:
            grad_output -- numpy array having the same shape as input

        Output:
            grad_input  -- numpy array has the same shape as grad_output. gradient w.r.t input
    '''
    def backward(self, grad_output):

        grad_input = np.copy(grad_output)

        grad_input[grad_input > 0] = 1

        return grad_input

'''
    CrossEntropyLossWithSoftmax

    Implementation of the combination of softmax function and cross entropy
    loss.  In classification tasks, we usually first apply the softmax function
    to map class-wise prediciton scores into a probability distribution over
    classes.  Then we use cross entropy loss to maximise the likelihood of
    the ground truth class's prediction.  Since softmax includes an exponential
    term and cross entropy includes a log term, we can simplify the formula by
    combining these two functions together, so that log and exp operations
    cancel out.  This way, we also avoid some precision loss due to floating
    point numerical computation.

    If we ignore the index on batch size and assume there is only one grouth
    truth per sample, the formula for softmax and cross entropy loss are:

        Softmax: prob[i] = exp(x[i]) / \sum_{j}exp(x[j])
        Cross_entropy_loss:  - 1 * log(prob[gt_class])

    Combining these two functions togther, we have:

        cross_entropy_with_softmax: -x[gt_class] + log(\sum_{j}exp(x[j]))

    In this assignment, you will implement both forward and backward
    computation.

    Arguments:
        None

    Reference: https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
'''
class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        pass

    def softmax(self, input):
        exps = np.exp(input)
        return exps / np.sum(exps, axis=1)[:,None]

    '''
        Forward computation of cross entropy with softmax. (3 points)

        You may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input    -- numpy array of shape (N, C), the prediction for each class, where C is number of classes
            gt_label -- numpy array of shape (N), it is an integer array and the value range from 0 to C-1 which
                        specify the ground truth class for each input
        Output:
            output   -- numpy array of shape (N), containing the cross entropy loss on each input
    '''
    def forward(self, input, gt_label):

        self.n, self.c = input.shape #number of classes

        self.gt_label = gt_label

        # n = gt_label.shape[0]

        self.probs = self.softmax(input)

        # print(self.gt_label.shape)
        # print(self.gt_label)


        output = - np.log(self.probs[range(self.n),gt_label])

        return output

    '''
        Backward computation of cross entropy with softmax. (3 points)

        It is recommended to resue the variable(s) in forward computation
        in order to simplify the formula.

        Arguments:
            grad_output -- numpy array of shape (N)

        Output:
            output   -- numpy array of shape (N, C), the gradient w.r.t input of forward function
    '''
    def backward(self, grad_output):

        grad_input = np.copy(self.probs)
        grad_input[range(self.n),self.gt_label] -= 1

        return grad_input

'''
    im2col (3 points)

    Consider 4 dimensional input tensor with shape (N, C, H, W), where:
        N is the batch dimension,
        C is the channel dimension, and
        H, W are the spatial dimensions.

    The im2col functions flattens each slidding kernel-sized block
    (C * kernel_h * kernel_w) on each spatial location, so that the output has
    the shape of (N, (C * kernel_h * kernel_w), out_H, out_W) and we can thus
    formuate the convolutional operation as matrix multiplication.

    The formula to compute out_H and out_W is the same as to compute the output
    spatial size of a convolutional layer.

    Arguments:
        input_data  -- numpy array of shape (N, C, H, W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- numpy array of shape (N, (C * kernel_h * kernel_w), out_H, out_W)
'''
def im2col(input_data, kernel_h, kernel_w, stride, padding):
    
    N, C, H, W = input_data.shape #(1, 3, 32, 32)
    # print(input_data.shape)
    # print(kernel_h, kernel_w, stride, padding) #2 2 2 1

    # out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1
    out_H = int((H + 2* padding - kernel_h)/stride) + 1
    out_W = int((W + 2* padding - kernel_w)/stride) + 1

    # print(out_H, out_W) # 17 17

    output_data = np.empty([N, (C * kernel_h * kernel_w), out_H, out_W])

    #Add padding to image
    input_data_with_padding = np.empty(shape=(N,C,H+2*padding, W+2*padding))
    for n in range(N):
        for c in range(C):
            input_data_with_padding[n,c] = pad_border(input_data[n,c], wx=padding, wy=padding)

    for n in range(N):
        for oH in range(out_H):
            for oW in range(out_W):

                #Translate point (oH, oW) to associated (y, x) position in original image

                # Alt 1. Kernel sliding centered at pixel
                y = stride*oH + int(kernel_h/2)
                x = stride*oW + int(kernel_w/2)

                # Alt 2. Kernel sliding with top left corner at pixel
                # y = stride*oH
                # x = stride*oW

                # build vector of length (C * kernel_h * kernel_w)
                flattened_slidding_kernel = get_flattened_slidding_kernel(input_data_with_padding, n, C, y, x, kernel_h, kernel_w)
                # print(flattened_slidding_kernel.shape)
                # print(output_data.shape)
                for index, c in enumerate(flattened_slidding_kernel):
                    output_data[n, index, oH, oW] = c

    return output_data



def get_flattened_slidding_kernel(input_data, n, n_of_channels, y, x, kernel_h, kernel_w):
    #Get region of interest

    # Alt 1. Kernel sliding centered at pixel
    roi = input_data[n, range(n_of_channels), y-int(kernel_h/2):y+int(np.ceil(kernel_h/2)), x-int(kernel_w/2): x + int(np.ceil(kernel_w/2))]

    # Alt 2. Kernel sliding with top left corner at pixel
    # roi = input_data[n, range(n_of_channels), y:y+kernel_h, x: x + kernel_w]
   
    # print("roi first pic first channel")
    # print(roi[0,1])

    #Flatten it
    # print("roi_flatten")
    roi_flatten = roi.flatten()
    # print(roi_flatten)
    return roi_flatten

def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img


'''
    col2im (3 points)

    Consider a 4 dimensional input tensor with shape:
        (N, (C * kernel_h * kernel_w), out_H, out_W)
    where:
        N is the batch dimension,
        C is the channel dimension,
        out_H, out_W are the spatial dimensions, and
        kernel_h and kernel_w are the specified kernel spatial dimension.

    The col2im function calculates each combined value in the resulting array
    by summing all values from the corresponding sliding kernel-sized block.
    With the same parameters, the output should have the same shape as
    input_data of im2col.  This function serves as an inverse subroutine of
    im2col, so that we can formuate the backward computation in convolutional
    layers as matrix multiplication.

    Arguments:
        input_data  -- numpy array of shape (N, (C * kernel_H * kernel_W), out_H, out_W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- output_array with shape (N, C, H, W)
'''
def col2im(input_data, kernel_h, kernel_w, stride=1, padding=0):
    ########################
    # TODO: YOUR CODE HERE #
    ########################

    return output_data

'''
    Conv2d

    Implementation of convolutional layer.  This layer performs convolution
    between each sliding kernel-sized block and convolutional kernel.  Unlike
    the convolution you implemented in HW1, where you needed flip the kernel,
    here the convolution operation can be simplified as cross-correlation (no
    need to flip the kernel).

    This layer has 2 learnable parameters, weight (convolutional kernel) and
    bias, which are specified and initalized in the init_param() function.
    You need to complete both forward and backward functions of the class.
    For backward, you need to compute the gradient w.r.t input, weight, and
    bias.  The input arguments: kernel_size, padding, and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You need to use im2col, col2im inside forward and backward respectively,
    which formulates the sliding window computation in a convolutional layer as
    matrix multiplication.

    Arguments:
        input_channel  -- integer, number of input channel which should be the same as channel numbers of filter or input array
        output_channel -- integer, number of output channel produced by convolution or the number of filters
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class Conv2d(object):
    def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
        self.output_channel = output_channel
        self.input_channel = input_channel
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
        self.bias = np.zeros(self.output_channel).astype(np.float32)

    '''
        Forward computation of convolutional layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, output_chanel, out_H, out_W)
    '''
    def forward(self, input):

        _,_,self.H, self.W = input.shape

        input_im2col = im2col(input, kernel_h=self.kernel_h, kernel_w=self.kernel_w, stride=self.stride, padding=self.padding)

        N, c_kh_kw, out_H, out_W = input_im2col.shape #(1, 27, 16, 16)
        
        #print(self.weight.shape) #(6, 3, 3, 3)

        output = np.empty([N, self.output_channel, out_H, out_W])

        #For each image
        for n in range(N):
            #For each filter/output channel
            for output_c in range(self.output_channel):
                #Apply the convolution (element_wise multiplication) over each pixel in the input,
                #suming over input channels
                for oH in range(out_H):
                    for oW in range(out_W):
                        output[n, output_c, oH, oW] = \
                            self.convolution_for_given_pixel(\
                                input_im2col[n,range(c_kh_kw),oH,oW], #input  
                                self.weight[output_c].flatten(), #weight
                                self.bias[output_c]) #bias

        return output

    def convolution_for_given_pixel(self,input,weights,bias):
        return np.sum(np.multiply(input, weights))+bias

    '''
        Backward computation of convolutional layer. (3 points)

        You need col2im and saved variables from forward() in this function.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
            grad_weight -- numpy array of shape(output_channel, input_channel, kernel_h, kernel_w), gradient w.r.t weight
            grad_bias   -- numpy array of shape(output_channel), gradient w.r.t bias
    '''

    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################

        N, output_channel, _,_ = grad_output.shape

        grad_input = np.empty(shape=(N, self.input_channel, self.H, self.W))
        grad_weight = np.empty(shape=(output_channel, self.input_channel, self.kernel_h, self.kernel_w))
        grad_bias = np.empty(shape=(output_channel))

        return grad_input, grad_weight, grad_bias

'''
    MaxPool2d

    Implementation of max pooling layer.  For each sliding kernel-sized block,
    maxpool2d computes the spatial maximum along each channels.  This layer has
    no learnable parameters.

    You need to complete both forward and backward functions of the layer.
    For backward, you need to compute the gradient w.r.t input.  Similar as
    conv2d, the input argument, kernel_size, padding and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You may use im2col, col2im inside forward and backward, respectively.

    Arguments:
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class MaxPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride

    '''
        Forward computation of max pooling layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, input_channel, out_H, out_W)



    '''
    def forward(self, input):

        _, self.input_channel, self.H, self.W = input.shape

        # print(input.shape) #(1, 3, 32, 32)

        input_im2col = im2col(input, kernel_h=self.kernel_h, kernel_w=self.kernel_w, stride=self.stride, padding=self.padding)

        N, c_kh_kw, out_H, out_W = input_im2col.shape #(1, 12, 16, 16)
        # print(input_im2col.shape)
        output = np.empty([N, self.input_channel, out_H, out_W])
        # print(output.shape) # 1 3 16 16

        for n in range(N):
            for channel in range(self.input_channel):
                for oH in range(out_H):
                    for oW in range(out_W):

                        range_where_to_look_in_im2col = \
                            range(channel*self.kernel_h*self.kernel_w, (channel+1)*self.kernel_h*self.kernel_w)                   


                        output[n,channel,oH, oW] = \
                            max(input_im2col[n, range_where_to_look_in_im2col, oH, oW])
        
        return output

    '''
        Backward computation of max pooling layer. (3 points)

        You should use col2im and saved variable(s) from forward().

        Arguments:
            grad_output -- numpy array of shape (N, input_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N, input_channel, out_H, out_W = grad_output.shape

        grad_input = np.empty(shape=(N, input_channel, self.H, self.W))

        #Incomplete...

        return grad_input
