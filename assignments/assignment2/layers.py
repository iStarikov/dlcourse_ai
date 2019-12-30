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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.zero_mask = None
        self.diff = None

    def forward(self, X):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        out = X.copy()
        self.zero_mask = X <= 0
        out[self.zero_mask] = 0
        return out
        # LReLu
        # relu[self.zero_mask] *= 0.01
        # self.zero_mask = (X > 0).astype(float)
        # return np.maximum(X, np.zeros_like(X))

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
        # print(f'd_out: {d_out}')
        d_input = d_out.copy()
        d_input[self.zero_mask] = 0
        # d_input[self.zero_mask] = 0.01
        # d_input = self.zero_mask * d_out
        return d_input

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
        full = np.dot(self.X, self.W.value) + self.B.value
        return full

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


class BatchNorm:
    def __init__(self, mean, sig):
        self.mean = mean
        self.sig = sig
        self.b = None
        self.V = None
        self.X = None

    def forward(self, X: np.ndarray):
        self.X = X
        self.mean = np.mean(X, axis=0)  # shape == (X.shape[1], )
        self.sig = np.sqrt(np.mean((X - self.mean) ** 2, axis=0) + 1e-7)
        out = (X - self.mean) / self.sig
        out = self.V * out + self.b

        return out

    def backward(self, d_out):
        d_input0 = self.V
        d_input1 = np.mean((self.X - self.mean), axis=0) / self.sig
        d_input = d_input0 * d_input1
        return d_input

    def params(self):
        return {'V': self.V,
                'B': self.b}
