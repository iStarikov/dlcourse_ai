import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg=0):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.n_input = n_input
        self.n_output = n_output
        self.h_size = hidden_layer_size
        # TODO Create necessary layers
        self.RL = ReLULayer()
        self.FC1 = FullyConnectedLayer(n_input=self.n_input, n_output=self.h_size)
        self.FC2 = FullyConnectedLayer(self.h_size, self.n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.FC1.W.grad = np.zeros_like(self.FC1.W.grad)
        self.FC1.B.grad = np.zeros_like(self.FC1.B.grad)
        self.FC2.W.grad = np.zeros_like(self.FC2.W.grad)
        self.FC2.B.grad = np.zeros_like(self.FC2.B.grad)
        # raise Exception("Not implemented!")

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        params = self.params()

        x = X.copy()
        x = self.FC1.forward(x)
        x = self.RL.forward(x)
        pred = self.FC2.forward(x)
        # print(f'SHAPE fc1: \n {np.sum(self.FC1.W.grad)}')
        # print(f'SHAPE b2: \n {np.sum(self.FC1.B.grad)}')

        loss, dpred = softmax_with_cross_entropy(pred, target_index=y)

        d_out = self.FC2.backward(dpred)
        d_out = self.RL.backward(d_out)
        grad = self.FC1.backward(d_out)

        # print(f'SHAPE fc1: \n {np.sum(self.FC1.W.grad)}')
        # print(f'SHAPE fc2: \n {np.sum(self.FC2.W.grad)}')

        if self.reg > 0:
            rloss_fc1, dW_rfc1 = l2_regularization(self.FC1.W.value, self.reg)
            rloss_fc2, dW_rfc2 = l2_regularization(self.FC2.W.value, self.reg)
            rloss_fc1B, dB_rfc1 = l2_regularization(self.FC1.B.value, self.reg)
            rloss_fc2B, dB_rfc2 = l2_regularization(self.FC2.B.value, self.reg)
            loss = loss + rloss_fc1 + rloss_fc2 + rloss_fc1B + rloss_fc2B
            self.FC1.W.grad += dW_rfc1
            self.FC2.W.grad += dW_rfc2
            self.FC1.B.grad += dB_rfc1
            self.FC2.B.grad += dB_rfc2

        # result = {'fc1_w': self.FC1.W.grad,
        #           'fc1_b': self.FC1.B.grad,
        #           'fc2_w': self.FC2.W.grad,
        #           'fc2_b': self.FC2.B.grad}
        return loss, grad

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        x = X.copy()

        x = self.FC1.forward(x)
        x = self.RL.forward(x)
        x = self.FC2.forward(x)

        y_hat = softmax(predictions=x)
        y_hat = np.argmax(y_hat, axis=1)
        return y_hat

    def params(self):
        result = {'FC1.W': self.FC1.W,
                  'FC1.B': self.FC1.B,
                  'FC2.W': self.FC2.W,
                  'FC2.B': self.FC2.B}
        return result
