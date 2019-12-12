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
    # TODO implement softmax
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


def softmax_with_cross_entropy(predictions: np.ndarray, target_index):
    '''
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
    '''
    pred_shape = predictions.shape
    if predictions.ndim == 1:
        predictions = predictions.reshape((-1, predictions.shape[0]))
    n_samples = predictions.shape[0]

    if not isinstance(target_index, np.ndarray):
        target_index = np.array(target_index)
    if target_index.ndim > 1:
        target_index = target_index.flatten()
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprobs = probs.copy()
    dprobs[range(n_samples), target_index] -= 1
    dprobs /= n_samples
    dprediction = dprobs.reshape(pred_shape)
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    # raise Exception("Not implemented!")

    # r*(x^2 + F) -> 2*r*x
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


class LinearSoftmaxClassifier:
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # print(batches_indices[epoch])
            # print(X.shape)
            loss_epoch = []
            n_batches = len(batches_indices)
            for batch in range(n_batches):
                X_train = X[batches_indices[batch]]
                y_train = y[batches_indices[batch]]
                loss, dW = linear_softmax(X_train, self.W, y_train)
                reg_loss, dW_reg = l2_regularization(self.W, reg)
                self.W -= learning_rate * (dW + dW_reg)
                loss = loss + reg_loss
                loss_epoch.append(loss)
            # end
            loss_epoch = np.sum(loss_epoch) / n_batches
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, loss: {loss_epoch:.5f}")

            loss_history.append(loss_epoch)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        # y_pred = np.zeros(X.shape[0], dtype=np.int)
        prediction = np.dot(X, self.W)
        y_pred = softmax(predictions=prediction)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
