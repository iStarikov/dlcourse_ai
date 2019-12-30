import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, softmax
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels, filter_size=3):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.input_shape = input_shape
        self.n_output_classes = n_output_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.filter_size = filter_size
        self.padding = 1

        c1 = int((input_shape[0] - self.filter_size + 2 * self.padding) / 1) + 1
        mp1 = int((c1 - 4) / 4) + 1
        c2 = int((mp1 - self.filter_size + 2 * self.padding) / 1) + 1
        self.size_after_2maxpool = int((c2 - 4) / 4) + 1

        self.RL1 = ReLULayer()
        self.RL2 = ReLULayer()
        self.MaxPool1 = MaxPoolingLayer(pool_size=4, stride=4)
        self.MaxPool2 = MaxPoolingLayer(pool_size=4, stride=4)
        self.Flatten = Flattener()
        self.Conv1 = ConvolutionalLayer(in_channels=self.input_shape[-1], out_channels=conv1_channels,
                                        filter_size=self.filter_size, padding=self.padding)
        self.Conv2 = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels,
                                        filter_size=self.filter_size, padding=self.padding)
        self.FC = FullyConnectedLayer(n_input=conv2_channels * self.size_after_2maxpool ** 2,
                                      n_output=self.n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        # self.Conv1.W.grad = np.zeros_like(self.Conv1.W.grad)
        # self.Conv1.B.grad = np.zeros_like(self.Conv1.B.grad)
        # self.Conv2.W.grad = np.zeros_like(self.Conv2.W.grad)
        # self.Conv2.B.grad = np.zeros_like(self.Conv2.B.grad)
        # self.FC.W.grad = np.zeros_like(self.FC.W.grad)
        # self.FC.B.grad = np.zeros_like(self.FC.B.grad)

        # Input -> Conv[3
        # x3] -> Relu -> Maxpool[4
        # x4] ->
        # Conv[3
        # x3] -> Relu -> MaxPool[4
        # x4] ->
        # Flatten -> FC -> Softmax

        x = self.Conv1.forward(X)
        x = self.RL1.forward(x)
        x = self.MaxPool1.forward(x)
        x = self.Conv2.forward(x)
        x = self.RL2.forward(x)
        x = self.MaxPool2.forward(x)
        x = self.Flatten.forward(x)
        pred = self.FC.forward(x)

        loss, dpred = softmax_with_cross_entropy(pred, target_index=y)

        d_out = self.FC.backward(dpred)
        d_out = self.Flatten.backward(d_out)
        d_out = self.MaxPool2.backward(d_out)
        d_out = self.RL2.backward(d_out)
        d_out = self.Conv2.backward(d_out)
        d_out = self.MaxPool1.backward(d_out)
        d_out = self.RL1.backward(d_out)
        _ = self.Conv1.backward(d_out)

        # param_ = self.Conv1.W
        # before_opt = param_.value[:2, :2]
        # print(f"PREDICT stage Conv1_W value: \n {before_opt} \n")
        # print(f"PREDICT stage Conv1_dW: \n {param_.grad[:2, :2]} \n")
        ## !! do not update params
        # print(f'SHAPE fc1: \n {np.sum(self.FC1.W.grad)}')
        # print(f'SHAPE fc2: \n {np.sum(self.FC2.W.grad)}')

        # result = {'fc1_w': self.FC1.W.grad,
        #           'fc1_b': self.FC1.B.grad,
        #           'fc2_w': self.FC2.W.grad,
        #           'fc2_b': self.FC2.B.grad}
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        x = self.Conv1.forward(X)
        x = self.RL1.forward(x)
        x = self.MaxPool1.forward(x)
        x = self.Conv2.forward(x)
        x = self.RL2.forward(x)
        x = self.MaxPool2.forward(x)
        x = self.Flatten.forward(x)
        x = self.FC.forward(x)

        y_hat = softmax(predictions=x)
        y_hat = np.argmax(y_hat, axis=1)
        return y_hat

    def params(self):
        result = {'Conv1.W': self.Conv1.W,
                  'Conv1.B': self.Conv1.B,
                  'Conv2.W': self.Conv2.W,
                  'Conv2.B': self.Conv2.B,
                  'FC.W': self.FC.W,
                  'FC.B': self.FC.B
                  }
        return result
