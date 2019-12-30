import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # Your final implementation shouldn't have any loops
    y_hat = predictions.copy()
    if y_hat.ndim == 1:
        y_hat -= np.max(y_hat)
    else:
        y_hat -= np.max(y_hat, axis=1, keepdims=True)
    exp = np.exp(y_hat)
    if y_hat.ndim == 1:
        probs = exp / np.sum(exp)  # sum(exp)
    else:
        probs = exp / np.sum(exp, axis=1, keepdims=True)  # sum(exp)

    # raise Exception("Not implemented!")
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if not isinstance(target_index, np.ndarray):
        target_index = np.array(target_index)
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        correct_logprobs = np.log(probs[range(probs.shape[0]), target_index])
        loss = -np.sum(correct_logprobs) / probs.shape[0]
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W ** 2)
    grad = 2.0 * reg_strength * W
    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    n_samples = X.shape[0]
    prediction = np.dot(X, W)  # N,M * M,C
    pred_shape = prediction.shape

    z = softmax(prediction)
    loss = cross_entropy_loss(z, target_index)
    # backprop
    dprobs = z.copy()
    dprobs[range(n_samples), target_index] -= 1
    dprobs /= n_samples
    dprobs = dprobs.reshape(pred_shape)
    dW = np.dot(X.T, dprobs)
    return loss, dW


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    pred_shape = preds.shape
    if preds.ndim == 1:
        preds = preds.reshape((-1, preds.shape[0]))
    n_samples = preds.shape[0]

    if not isinstance(target_index, np.ndarray):
        target_index = np.array(target_index)
    if target_index.ndim > 1:
        target_index = target_index.flatten()
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)

    dprobs = probs.copy()
    dprobs[range(n_samples), target_index] -= 1
    dprobs /= n_samples
    d_preds = dprobs.reshape(pred_shape)
    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.zero_mask = None

    def forward(self, X):
        self.zero_mask = (X > 0).astype(float)
        return np.maximum(X, np.zeros_like(X))
        ## LReLu
        ## relu[self.zero_mask] *= 0.01

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return self.zero_mask * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:

    def __init__(self, n_input, n_output):
        # a = 0.001
        a = 1 / np.sqrt(n_input / 2)
        self.W = Param(a * np.random.randn(n_input, n_output))
        # np.random.seed(0)
        self.B = Param(a * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W,
                'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''
        self.a = 0.01
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = 1
        self.W = Param(self.a * np.random.randn(filter_size, filter_size,
                                                in_channels, out_channels))
        # self.B = Param(self.a * np.random.randn(1, out_channels))
        self.B = Param(np.zeros(out_channels))
        self.X_cache = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = int((height + 2 * self.padding - self.filter_size) / self.stride) + 1
        out_width = int((width + 2 * self.padding - self.filter_size) / self.stride) + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        self.X = X.copy()
        pad = np.zeros((batch_size,
                        height + 2 * self.padding,
                        width + 2 * self.padding,
                        channels))
        pad[:,
        self.padding:height + self.padding,
        self.padding:width + self.padding,
        :] = self.X

        self.X_cache = (X, pad)

        # for i in range(out_height):
        #     for j in range(out_width):
        #         win = pad[:,
        #                   i * self.stride:i * self.stride + self.filter_size,
        #                   j * self.stride:j * self.stride + self.filter_size,
        #                   :]
        #         win2d = win.reshape(batch_size, -1)
        #         W2d = self.W.value.reshape(-1, self.out_channels)
        #         out[:, i, j, :] = np.dot(win2d, W2d) + self.B.value

        pad = pad[:, :, :, :, np.newaxis]

        for y in range(out_height):
            for x in range(out_width):
                X_slice = pad[:, y:y + self.filter_size, x:x + self.filter_size, :, :]
                out[:, y, x, :] = np.sum(X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value

        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        X, pad = self.X_cache

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        #
        # print(f'd_out shape {d_out.shape}')
        # print(f'X shape {self.X.shape}')

        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        d_pad = np.zeros_like(pad)

        # Try to avoid having any other loops here too
        for i in range(out_height):
            for j in range(out_width):
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                # win = pad[:,
                #           i * self.stride:i * self.stride + self.filter_size,
                #           j * self.stride:j * self.stride + self.filter_size,
                #           :]
                # win2d = win.reshape(batch_size, -1)
                # W2d = self.W.value.reshape(-1, self.out_channels)
                #
                # W2dgrad = np.dot(win2d.T, d_out[:, i, j, :])
                # self.W.grad += W2dgrad.reshape((self.filter_size, self.filter_size, self.in_channels, self.out_channels))
                # self.B.grad += np.sum(d_out[:, i, j, :], axis=0, keepdims=False)
                #
                # d_input_win = np.dot(d_out[:, i, j, :], W2d.T)
                # d_pad[:,
                #       i * self.stride:i * self.stride + self.filter_size,
                #       j * self.stride:j * self.stride + self.filter_size,
                #       :] += d_input_win.reshape((batch_size, self.filter_size, self.filter_size, channels))

                X_slice = pad[:, i:i + self.filter_size, j:j + self.filter_size, :, np.newaxis]
                grad = d_out[:, i, j, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * X_slice, axis=0)

                d_pad[:, i:i + self.filter_size, j:j + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)

            self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        d_pad = d_pad[:,
                self.padding:height + self.padding,
                self.padding:width + self.padding,
                :]

        return d_pad

    # def mask_gen(self, tensor, row_idx, col_idx):
    #     return tensor[range(tensor.shape[0]),
    #                     i * self.stride:i * self.stride + self.filter_size,
    #                     j * self.stride:j * self.stride + self.filter_size,
    #                     :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.max_mask = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        x_in = self.X
        self.max_mask = np.zeros_like(x_in)

        for i in range(out_height):
            for j in range(out_width):
                win = x_in[:,
                      i * self.stride:i * self.stride + self.pool_size,
                      j * self.stride:j * self.stride + self.pool_size,
                      :]
                # print(win)
                # max values per image for each kernel and each sample
                max_value = np.max(np.max(win, axis=1, keepdims=True), axis=2, keepdims=True)
                # print(max_value.shape)
                # print((max_value == win).shape)
                out[:, i:i + 1, j:j + 1, :] = max_value
                self.max_mask[:,
                i * self.stride:i * self.stride + self.pool_size,
                j * self.stride:j * self.stride + self.pool_size,
                :] = np.isclose(win, max_value).astype(int)
                # np.isclose(win, max_value).astype(int)  # problem here
                # np.isclose(win, np.repeat(np.repeat(max_value, 2, axis=1), 2, axis=1)).astype(int)
        # print(f' OUT: {out.shape}')
        # print(self.max_mask)
        return out

    def backward(self, d_out):
        # print('BACKWARD')
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, height_out, width_out, _ = d_out.shape
        # print(f"SHAPE max_pool mask: \n {self.max_mask.shape}")

        zero_input = np.zeros_like(self.max_mask)
        repeats = np.floor(height / height_out)
        d_out = np.repeat(np.repeat(d_out, repeats, axis=1), repeats, axis=2)
        _, height_rep, width_rep, _ = d_out.shape
        zero_input[0:batch_size, 0:height_rep, 0:width_rep, 0:channels] = d_out
        # print(f"SHAPE d_out after repeat: \n {zero_input.shape}")
        return self.max_mask * zero_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        batch_size, *_ = X.shape
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
