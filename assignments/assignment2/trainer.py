from copy import deepcopy

import numpy as np

from metrics import multiclass_accuracy


class Dataset:
    """
    Utility class to hold training and validation data
    """

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-2,
                 learning_rate_decay=1.0):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        """
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay

        self.optimizers = None

    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy(self, X, y):
        """
        Computes accuracy on provided data using mini-batches
        """
        indices = np.arange(X.shape[0])
        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)

        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            pred[batch_indices] = pred_batch

        # param_ = self.model.FC1.W
        # before_opt = param_.value[:3, :5]
        # print(f"PREDICT stage FC1_W value: \n {before_opt} \n")
        # print(f"PREDICT stage FC1_dW: \n {param_.grad[:3, :5]}")
        # param_ = self.model.params()['FC1_W']
        # print(f"PREDICT stage FC1_dW from PARAMS: \n {param_.grad[:3, :5]} \n")  # grad are same

        # param_ = self.params()['FC1_W']
        # before_opt = param_.value[:3, :5]
        # print(f"PREDICT stage FC1_W value from PARAMS: \n {before_opt} \n")  # the same as from model
        # print(np.unique(y, return_counts=True))
        # print(np.unique(pred, return_counts=True))
        # print(pred)

        return multiclass_accuracy(pred, y)

    def fit(self):
        """
        Trains a model
        """
        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0]

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for epoch in range(self.num_epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = []
            # print(f'\n EPOCH: {epoch}')
            for batch_n, batch_indices in enumerate(batches_indices):
                # TODO Generate batches based on batch_indices and
                # use model to generate loss and gradients for all
                # the params
                train_X = self.dataset.train_X[batch_indices]
                train_y = self.dataset.train_y[batch_indices]

                loss, grad = self.model.compute_loss_and_gradients(train_X, train_y)

                # if (epoch % 3 == 0) & (batch_n % 200 == 0):
                #     print(f'\n BATCH: {batch_n}')
                #     param_ = self.model.params()['FC1_W']
                #     print('Before optim from model PARAMS')
                #     before_opt = param_.value[:3, :5]
                #     print(f"FC1_W value: \n {before_opt}")
                #     print(f"FC1_dW: \n {param_.grad[:3, :5]}")
                #     # print(f"From model FC1_W value: \n {self.model.FC1.W.value[:3, :5]}")  # the same as from params

                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)

                # if (epoch % 3 == 0) & (batch_n % 200 == 0):
                #     param_ = self.model.params()['FC1_W']
                #     print('After optim from model PARAMS')
                #     after_opt = param_.value[:3, :5]
                #     print(f"FC1_W value: \n {after_opt}")  # params updated in model and in params
                #     print(f"FC1_dW: \n {param_.grad[:3, :5]} \n \n")
                # print(f"From model FC1_W value: \n {self.model.FC1.W.value[:3, :5]}")  # the same as from params

                # if (epoch % 3 == 0) & (batch_n == 3):
                #     if param_name == 'FC1_W':
                #         print('after optim')
                #         print(param.value[:3, :5])
                #         print('model param value')
                #         print(self.model.params()[param_name].value[:3, :5])
                #         print(np.all(self.model.params()[param_name].value[:3, :5] == param.value[:3, :5]))
                #         print(np.all(before_opt == param.value[:3, :5]))

                batch_losses.append(loss)

            # classic decay
            # if (epoch % 10 == 0) & epoch != 0:
            #     if np.not_equal(self.learning_rate_decay, 1.0):
            #         self.learning_rate *= 1/(1 + self.learning_rate_decay*epoch)

            if np.not_equal(self.learning_rate_decay, 1.0):
                self.learning_rate *= self.learning_rate_decay

            ave_loss = np.mean(batch_losses)
            # print('PREDICT STAGE')
            # print('train predict \n')
            train_accuracy = self.compute_accuracy(self.dataset.train_X,
                                                   self.dataset.train_y)
            # print('val predict \n')
            val_accuracy = self.compute_accuracy(self.dataset.val_X,
                                                 self.dataset.val_y)
            if epoch % 5 == 0:
                print(f"Epoch: {epoch}, Loss:{batch_losses[-1]:.6f}, "
                      f"Train accuracy: {train_accuracy:.6f}, val accuracy: {val_accuracy:.6f}")

            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

        return loss_history, train_acc_history, val_acc_history
