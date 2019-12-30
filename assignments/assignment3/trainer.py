from copy import deepcopy

import numpy as np

from metrics import multiclass_accuracy


class Dataset:
    ''' 
    Utility class to hold training and validation data
    '''

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y

        
class Trainer:
    '''
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    '''
    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-3,
                 learning_rate_decay=1.0):
        '''
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
        '''
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
        '''
        Computes accuracy on provided data using mini-batches
        '''
        indices = np.arange(X.shape[0])

        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)

        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            pred[batch_indices] = pred_batch
        # print(f"\n prediction {np.unique(pred, return_counts=True)}")
        # print(f"ground through {np.unique(y, return_counts=True)} \n")

        # param_ = self.model.Conv1.W
        # before_opt = param_.value[:2, :2]
        # print(f"PREDICT stage Conv1_W value: \n {before_opt} \n")
        # print(f"PREDICT stage Conv1_dW: \n {param_.grad[:2, :2]} \n")

        return multiclass_accuracy(pred, y)
        
    def fit(self):
        '''
        Trains a model
        '''
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

            for batch_indices in batches_indices:
                batch_X = self.dataset.train_X[batch_indices]
                batch_y = self.dataset.train_y[batch_indices]

                loss = self.model.compute_loss_and_gradients(batch_X, batch_y)
                
                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    value1 = param.value
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)
                    # if param_name == 'Conv1.W':
                    #     print(f'{param_name.upper()}')
                    #     # print(f'{value1}')
                    #     # print(f'{param.value}')
                    #     update_optimizer = np.mean(value1 - param.value)
                    #     update_explicit = np.mean(param.grad * self.learning_rate)
                    #     print(f'FROM PARAMS dict '
                    #           f'\n  optimizer updates: {update_optimizer}')
                    #     print(f'IS model and optim param value equal: {np.all(np.isclose(param.value, self.model.params()[param_name].value))}')
                    #     print(f"  GRAD for param value: {np.mean(param.grad)}, ")
                    #     print(f"\n  explicit SGD update: {update_explicit} \n")
                    #     print(f"  IS optimizer on params works: {np.isclose(update_explicit, update_optimizer)} \n")
                batch_losses.append(loss)

            self.learning_rate *= self.learning_rate_decay
            ave_loss = np.mean(batch_losses)
            train_accuracy = self.compute_accuracy(self.dataset.train_X,
                                                   self.dataset.train_y)
            val_accuracy = self.compute_accuracy(self.dataset.val_X,
                                                 self.dataset.val_y)
            if epoch % 4 == 0:
                print(f"EPOCH {epoch}: Loss: {batch_losses[-1]}, "
                      f"Train accuracy: {train_accuracy}, val accuracy: {val_accuracy}")
            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)
        return loss_history, train_acc_history, val_acc_history
