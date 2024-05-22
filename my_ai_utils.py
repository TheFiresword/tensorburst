import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
import pickle
import seaborn as sn
from tqdm import tqdm
from enum import Enum
import copy

class Color(Enum): # Colors code to display metrics
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    reset = "\033[0m"
    
    def __str__(self) -> str:
        return str(self.value)

class Usage(Enum): # NN usages
    logisticRegression = "logisticRegression"
    multiClassification = "multiClassification"
    regression = "regression"
    
class Processing():
    def __init__(self) -> None:
        pass
    
    def one_hot_encode(data: list | np.ndarray):
        unique_data = np.unique(data)
        num_unique = len(unique_data)
        one_hot_vectors = np.zeros((len(data), num_unique), dtype=int)
        indexer = np.arange(len(data))
        one_hot_vectors[indexer, np.searchsorted(unique_data, data)] = 1
        return one_hot_vectors
    
    def one_hot_decode(data: np.ndarray, classes: list = None) -> np.ndarray:
        if not classes: classes = range(data.shape[-1])
        decoded_vectors = np.argmax(data, axis=-1)
        mapping_function = np.vectorize(lambda index : classes[index])
        return mapping_function(decoded_vectors)        
            
    def arange(data: list | np.ndarray):
        if type(data) is list: data = np.array(data)
        if len(data.shape) == 1: data = np.reshape(data, (data.shape[0], 1))
        return data
    
    def shuffle_dataset(dataset_X : np.ndarray, dataset_Y : np.ndarray) -> tuple[np.ndarray]:
        dataset_X, dataset_Y = Processing.arange(dataset_X), Processing.arange(dataset_Y)
        permutation = np.random.permutation(len(dataset_X))    
        shuffled_X, shuffled_Y = dataset_X[permutation], dataset_Y[permutation]
        return shuffled_X, shuffled_Y
    
    def train_test_split(dataset_X : np.ndarray, dataset_Y : np.ndarray, test_size:float) -> tuple[tuple[np.ndarray]]:
        X, Y = Processing.shuffle_dataset(dataset_X, dataset_Y)
        index = int(len(dataset_X)*test_size)
        test = X[:index+1], Y[:index+1]
        train = X[index+1:], Y[index+1:]
        return train, test
    
    def KFold(dataset_X : np.ndarray, dataset_Y : np.ndarray, k : int) -> list[tuple[tuple[np.ndarray]]]:
        X, Y = Processing.shuffle_dataset(dataset_X, dataset_Y)
        step = len(X) // k
        kfolds = []        
        for i in range(k):
            start, end = i * step, (i + 1) * step
            test_X, test_Y = X[start:end], Y[start:end]
            train_X, train_Y = np.vstack((X[:start], X[end:])), np.vstack((Y[:start], Y[end:]))
            kfolds.append(((train_X, train_Y), (test_X, test_Y)))        
        return kfolds
    
'''
# Test Processing
data = np.array([
    [1, 0, 0], [0.2, 0.1, 0.7], [0.95, 0.09, 0.001], [1, 0, 0], [0.2, 0.1, 0.7], [0.95, 0.09, 0.001], [0.95, 0.09, 0.001]
    ])
# np.random.rand(7,)
other_data = np.array([1, 2, 3, 4, 5, 6, 7])
print(data.shape, other_data.shape)
for (train, test) in Processing.KFold(data, other_data, 3):
    print("Train X: ",train[0].shape,"Train Y: ",train[1].shape, "Test X: ", test[0].shape, "Test Y: ",test[1].shape)
    print(test[1])

#print(data.shape, other_data.shape)
#x, y = Processing.shuffle_dataset(data, other_data)
#print(x.shape, y.shape, y)
#classes = ['j', 'm', 'k']
#z = Processing.one_hot_decode(data, classes)
#print(z.shape, z)
'''

class Metric:
    def __init__(self, *args, **kwargs):    
        self.output = None
        if args:
            return self.__call__(*args, **kwargs)
        return
        
    def __call__(self, predictions:np.ndarray, labels:np.ndarray, 
                 usage : Usage = Usage.logisticRegression):        
        if usage == Usage.logisticRegression: predictions = np.rint(predictions)
        if usage == Usage.multiClassification: # Predictions are softmax outputs
            predictions, labels = Processing.one_hot_decode(predictions), Processing.one_hot_decode(labels)
        predictions, labels = predictions.flatten(), labels.flatten()
        return predictions, labels
    
class Accuracy(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, predictions:np.ndarray, labels:np.ndarray, 
                 usage : Usage = Usage.logisticRegression, verbose:bool=True):
        predictions, labels = super().__call__(predictions, labels, usage)
        self.output = np.equal(predictions, labels).sum() / labels.shape[0]
        if verbose: print(f" {Color.red if self.output < 0.5 else Color.green}accuracy: {self.output} {Color.reset} ")
        return self.output     

class ConfusionMatrix(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, predictions:np.ndarray, labels:np.ndarray, classes : list,
                 usage : Usage = Usage.multiClassification, verbose:bool=True):
        # Only single label
        predictions, labels = super().__call__(predictions, labels, usage)
        "positive|negative" + "true|false"
        matrix = np.zeros((len(classes), len(classes)))
        for l, p in zip(labels, predictions):
            matrix[p, l] += 1
        for l in np.unique(labels):
            matrix[:,l] /= np.count_nonzero(labels==l)
        plt.figure()
        sn.heatmap(matrix, vmin=0.00, vmax=1.00, cmap=sn.color_palette("Blues", as_cmap=True), 
                   annot=True, fmt=".2f", mask= matrix <= 0.01, xticklabels=classes, yticklabels=classes)
        if verbose: plt.show()
        self.output = matrix
        return matrix

'''
# Metrics Test
print(Color.red)
predictions = np.array([
    [1, 0, 0], [0.2, 0.1, 0.7], [0.95, 0.09, 0.001],
    [0.66, 0.32, 0.01], [0.2, 0.1, 0.7], [1, 0, 0] 
])
truth = np.array([
    [1, 0, 0], [1, 0, 0], [0, 1, 0],
    [1, 0, 0], [0, 0, 1], [0, 1, 0]
])
a = Accuracy(predictions, truth, usage=Usage.multiClassification, verbose=True)
b = ConfusionMatrix(predictions, truth, usage=Usage.multiClassification, verbose=True, classes=['a', 'b', 'c'])
print(a.output, b.output)
'''

class Recall(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, predictions:np.ndarray, labels:np.ndarray,
                 usage : Usage = Usage.logisticRegression, verbose:bool=True):
        #print(f"{self.red if recall < 0.50 else self.green} Recall: {recall} {self.reset}")
        return

class F1(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, predictions:np.ndarray, labels:np.ndarray,
                 usage : Usage = Usage.logisticRegression, verbose:bool=True):
        # print(f"{self.blue} f1 score: {f1_score} {self.reset} ")
        return

class Precision(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, predictions:np.ndarray, labels:np.ndarray,
                 usage : Usage = Usage.logisticRegression, verbose:bool=True):
        # print(f" {self.red if precision < 50 else self.green} Precision: {precision} {self.reset}")
        return

class AUC(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, predictions:np.ndarray, labels:np.ndarray,
                 usage : Usage = Usage.logisticRegression, verbose:bool=True):
        # print(f"{self.cyan} auc surface: {au_roc} {self.reset}")
        return


class Regularizer:
    def __init__(self, name:str=None, alpha:float=0, beta:float=0) -> None:
        self.name = name
        (self.alpha, self.beta) = (alpha, beta)
        self.gradient_regularization = 0
        self.loss_regularization = 0
    
    def loss_reg(self, all_kernels: np.array) -> float:
        all_kernels = np.concatenate([kernel.ravel() for kernel in all_kernels])
        match self.name:
            case 'ridge':
                self.loss_regularization = self.alpha*np.square(all_kernels).sum()                
            case 'lasso':
                self.loss_regularization = self.alpha*np.absolute(all_kernels).sum()
            case 'elastic-net':
                self.loss_regularization = self.beta*(self.alpha*np.absolute(all_kernels).sum() + 
                                                      (1-self.alpha)*np.square(all_kernels).sum())
            case 'max_norm':
                return
        return self.loss_regularization
    
    def gradients_reg(self, kernel: np.array) -> np.array:
        match self.name:
            case 'ridge':
                self.gradient_regularization = 2*self.alpha*kernel
            case 'lasso':
                self.gradient_regularization = self.alpha*np.sign(kernel)
            case 'elastic-net':
                self.gradient_regularization = self.beta*((1-self.alpha)*kernel + self.alpha*np.sign(kernel))
        return self.gradient_regularization


class Loss:
    def __init__(self, loss_function="l2") -> None:
        self.loss_fn = loss_function
        self.error = None
        self.dLoss_per_prediction = None
    
    def __call__(self, y_predicted: np.ndarray, y_recorded: np.ndarray) -> tuple[float, np.ndarray]:
        match self.loss_fn:
            case "l0-1":
                self.dLoss_per_prediction = 0
                self.error =  np.where(y_recorded != y_predicted, 1, 0).sum(axis=-1)
            case "hinge":
                self.dLoss_per_prediction = np.where(1-y_recorded*y_predicted >=0, -y_recorded, 0)
                self.error = np.maximum(0, 1-y_recorded*y_predicted).sum(axis=-1)            
            case "l2" | "quadratic":
                self.dLoss_per_prediction = 2*(y_predicted-y_recorded)
                self.error = np.square(y_predicted-y_recorded).sum(axis=-1)
            case "l1":
                self.dLoss_per_prediction = np.sign(y_predicted)
                self.error = np.absolute(y_recorded-y_predicted).sum(axis=-1)
            case "binary_cross_entropy":
                assert len(y_recorded.shape) >= 2 and y_recorded.shape[-1] == 1
                epsilon = 1e-7
                # Clipping y_predicted to avoid log(0)
                y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
                self.dLoss_per_prediction = (y_predicted-y_recorded) / (y_predicted * (1- y_predicted))
                self.error = - (y_recorded * np.log(y_predicted) + (1 - y_recorded) * np.log(1 - y_predicted)).sum(axis=-1)               
            case "categorical_cross_entropy":
                assert len(y_recorded.shape) >= 2 and y_recorded.shape[-1] > 1
                assert np.min(y_predicted) >= 0 and np.max(y_predicted) <= 1
                epsilon = 1e-7
                y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
                self.dLoss_per_prediction = (y_predicted - y_recorded) / y_recorded.shape[-1]
                self.error = (-y_recorded * np.log(y_predicted)).sum(axis=-1)
        self.error = np.mean(self.error, axis=0)
        return self.error, self.dLoss_per_prediction


class LearningRateScheduler:
    def __init__(self, initial_lr: float) -> None:
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.history = []        
    
    def __call__(self, step: int = None):
        return self.lr
    
    def set_steps_per_epoch(self, n : int):
        self.num_steps_per_epoch = n
    
    def set_decays_step(self, n: int):
        self.decay_steps = n
        
    def evolution(self):
        plt.plot(self.history)
        plt.show()

class StepDecay(LearningRateScheduler):
    def __init__(self, initial_lr: float, decay_rate : float, epochs_drop : int = 1, num_steps_per_epoch : int = 1,
                 warmup : bool = False, warmup_steps : int = 0, hold_steps : int = 0) -> None:
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.epochs_drop = epochs_drop
        self.num_steps_per_epoch = num_steps_per_epoch
        self.warmup = warmup
        self.hold_steps = hold_steps
        self.warmup_steps = warmup_steps
        if warmup: self.warmup_scheduler = Warmup(initial_lr, warmup_steps)
        
    def __call__(self, step:int):
        if self.warmup and step <= self.warmup_steps + self.hold_steps:
            self.lr = self.warmup_scheduler(step)
        else:
            step -= (self.warmup_steps + self.hold_steps)
            epoch = step // self.num_steps_per_epoch
            self.lr = self.initial_lr * self.decay_rate**(epoch // self.epochs_drop)
        self.history.append(self.lr)
        return self.lr

class CosineDecay(LearningRateScheduler):
    def __init__(self, initial_lr: float, decay_steps: int = 0, alpha : float = 0.0,
                 warmup : bool = False, warmup_steps : int = 0, hold_steps : int = 0) -> None:
        super().__init__(initial_lr)
        self.alpha = alpha
        self.decay_steps = decay_steps - warmup_steps - hold_steps
        self.warmup = warmup
        self.hold_steps = hold_steps
        self.warmup_steps = warmup_steps
        if warmup: self.warmup_scheduler = Warmup(initial_lr, warmup_steps)
        
    def __call__(self, step:int) -> float:
        if self.warmup and step <= self.warmup_steps + self.hold_steps:
            self.lr = self.warmup_scheduler(step)
        else:
            step -= (self.warmup_steps + self.hold_steps)
            step = min(step, self.decay_steps) 
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
            self.lr = self.initial_lr * (self.alpha + (1-self.alpha)*cosine_decay)
        self.history.append(self.lr)
        return self.lr 
     
class ExponentialDecay(LearningRateScheduler):
    def __init__(self, initial_lr: float, decay_rate: float, num_steps_per_epoch : int = 1, 
                 warmup : bool = False, warmup_steps : int = 0, hold_steps : int = 0) -> None:
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.num_steps_per_epoch = num_steps_per_epoch
        self.warmup = warmup
        self.hold_steps = hold_steps
        self.warmup_steps = warmup_steps
        if warmup: self.warmup_scheduler = Warmup(initial_lr, warmup_steps)
                
    def __call__(self, step : int) -> float :
        if self.warmup and step <= self.warmup_steps + self.hold_steps:
            self.lr = self.warmup_scheduler(step)
        else:
            step -= (self.warmup_steps + self.hold_steps)
            epoch = step // self.num_steps_per_epoch
            self.lr = self.initial_lr * math.exp(-self.decay_rate*epoch)
        self.history.append(self.lr)
        return self.lr

class Warmup(LearningRateScheduler):
    def __init__(self, target_lr : float, warm_steps : int, initial_lr: float=0.0) -> None:
        super().__init__(initial_lr)
        self.target_lr = target_lr
        self.warm_steps = warm_steps
    
    def __call__(self, step : int) -> float:
        if step <= self.warm_steps: self.lr = min(self.target_lr, self.initial_lr + step * (self.target_lr-self.initial_lr) / self.warm_steps)
        self.history.append(self.lr)
        return self.lr

'''
# Test LR Schedulers
steps_per_epoch = 2
a=ExponentialDecay(initial_lr=0.01, decay_rate=0.4, num_steps_per_epoch=steps_per_epoch, warmup=True, warmup_steps=steps_per_epoch, hold_steps=0)
b= Warmup(target_lr=0.01, warm_steps=steps_per_epoch)
c = StepDecay(initial_lr=0.01, decay_rate=0.4, epochs_drop=1, num_steps_per_epoch=steps_per_epoch, warmup=True, warmup_steps=steps_per_epoch)
d = CosineDecay(initial_lr=0.01, decay_steps=steps_per_epoch*5, alpha=0.0, warmup=True, warmup_steps=steps_per_epoch, hold_steps=steps_per_epoch)
for i in range(steps_per_epoch*5): 
    d(i), c(i), a(i), b(i)
d.evolution()
'''

class Optimizer:
    def __init__(self, lr : float | LearningRateScheduler=0.01, epsilon:float=1e-7) -> None:
        if not isinstance(lr, LearningRateScheduler): lr = LearningRateScheduler(lr)
        self.lr = lr
        self.epsilon = epsilon
        
    def get_gradients(self, nn_cache: list[dict], num_layer: int, epoch: int = None)->np.ndarray:
        return nn_cache[epoch]['gradients'][num_layer]
    
    def __call__(self, nn_cache: list[dict], num_layer: int, step : int)->np.ndarray:
        gradient_W, gradient_biases = self.get_gradients(nn_cache, num_layer, len(nn_cache)-1)
        return self.lr(step) * gradient_W, self.lr(step) * gradient_biases

    def show_lr_evolution(self):
        self.lr.evolution()
        
class Momentum(Optimizer):
    def __init__(self, alpha: float=0.9, lr: float | LearningRateScheduler=0.01) -> None:
        super().__init__(lr)
        self.alpha = alpha
        self.previous_updates = {}
        
    def __call__(self, nn_cache: list[dict], num_layer: int, step : int):        
        (gradient_W, gradient_biases) = self.get_gradients(nn_cache, num_layer, len(nn_cache)-1)
        if num_layer not in self.previous_updates:
            self.previous_updates[num_layer] = {
                'W': np.zeros_like(gradient_W),
                'biases': np.zeros_like(gradient_biases)
            }
        previous_update = self.previous_updates[num_layer]
        gradient_W = self.alpha * previous_update['W'] + self.lr(step) * gradient_W
        gradient_biases = self.alpha * previous_update['biases'] + self.lr(step) * gradient_biases
        self.previous_updates[num_layer]['W'], self.previous_updates[num_layer]['biases'] = gradient_W, gradient_biases
        return gradient_W, gradient_biases
    
class Adam(Optimizer):
    def __init__(self, beta_1: float = 0.9, beta_2: float=0.999, lr : float| LearningRateScheduler = 0.01, epsilon:float=1e-7) -> None:
        super().__init__(lr, epsilon)
        self.beta_1, self.beta_2 = beta_1, beta_2
        self.square_sums = {}
        self.linear_sums = {}
        
    def __call__(self, nn_cache: list[dict], num_layer: int, step : int):
        epoch = len(nn_cache)-1
        (gradient_W, gradient_biases) = self.get_gradients(nn_cache, num_layer, epoch)
        if num_layer not in self.square_sums:
            self.square_sums[num_layer] = {
                'W': np.zeros_like(gradient_W),
                'biases': np.zeros_like(gradient_biases)
            }
        if num_layer not in self.linear_sums:
            self.linear_sums[num_layer] = {
                'W': np.zeros_like(gradient_W),
                'biases': np.zeros_like(gradient_biases)
            }
        self.square_sums[num_layer]['W'] = self.beta_2*self.square_sums[num_layer]['W'] + \
            (1-self.beta_2)*np.square(gradient_W)
        self.square_sums[num_layer]['biases'] = self.beta_2*self.square_sums[num_layer]['biases'] + \
            (1-self.beta_2)*np.square(gradient_biases)
        
        self.linear_sums[num_layer]['W'] = self.beta_1*self.linear_sums[num_layer]['W'] + \
            (1-self.beta_1)*gradient_W
        self.linear_sums[num_layer]['biases'] = self.beta_1*self.linear_sums[num_layer]['biases'] + \
            (1-self.beta_1)*gradient_biases
        
        square_sum, linear_sum = self.square_sums[num_layer], self.linear_sums[num_layer]
        
        first_momentum = ((linear_sum['W'] / (1 - self.beta_1**epoch)), (linear_sum['biases'] / (1 - self.beta_1**epoch)))
        
        second_momentum = ((square_sum['W'] / (1 - self.beta_2**epoch)), (square_sum['biases'] / (1 - self.beta_2**epoch)))
        
        gradient_W = self.lr(step) * first_momentum[0] / (self.epsilon + np.sqrt(second_momentum[0]))
        gradient_biases = self.lr(step) * first_momentum[1] / (self.epsilon + np.sqrt(second_momentum[1]))
        return gradient_W, gradient_biases
    
class Adagrad(Optimizer):
    def __init__(self, lr: float=0.01, epsilon:float=1e-7) -> None:
        super().__init__(lr, epsilon)
        self.square_sums = {}
        
    def __call__(self, nn_cache: list[dict], num_layer: int, step : int):
        (gradient_W, gradient_biases) = self.get_gradients(nn_cache, num_layer, len(nn_cache)-1)
        if num_layer not in self.square_sums:
            self.square_sums[num_layer] = {
                'W': np.zeros_like(gradient_W),
                'biases': np.zeros_like(gradient_biases)
            }
        self.square_sums[num_layer]['W'] += np.square(gradient_W)
        self.square_sums[num_layer]['biases'] += np.square(gradient_biases)
        square_sum = self.square_sums[num_layer]
        gradient_W = gradient_W*self.lr(step)/(self.epsilon + np.sqrt(square_sum['W']))
        gradient_biases = gradient_biases*self.lr(step)/(self.epsilon + np.sqrt(square_sum['biases']))        
        return gradient_W, gradient_biases
    
class RMSprop(Optimizer):
    def __init__(self, beta: float=0.9, lr: float=0.01, epsilon:float=1e-7) -> None:
        super().__init__(lr, epsilon)
        self.beta = beta
        self.square_sums = {}

    def __call__(self, nn_cache: list[dict], num_layer: int, step : int):
        (gradient_W, gradient_biases) = self.get_gradients(nn_cache, num_layer, len(nn_cache)-1)
        if num_layer not in self.square_sums:
            self.square_sums[num_layer] = {
                'W': np.zeros_like(gradient_W),
                'biases': np.zeros_like(gradient_biases)
            }
        self.square_sums[num_layer]['W'] = self.beta*self.square_sums[num_layer]['W'] + (1-self.beta)*np.square(gradient_W)
        self.square_sums[num_layer]['biases'] = self.beta*self.square_sums[num_layer]['biases'] + (1-self.beta)*np.square(gradient_biases)
        square_sum = self.square_sums[num_layer]
        gradient_W = gradient_W*self.lr(step)/(self.epsilon + np.sqrt(square_sum['W']))
        gradient_biases = gradient_biases*self.lr(step)/(self.epsilon + np.sqrt(square_sum['biases']))        
        return gradient_W, gradient_biases

   
class Activation:
    def relu(x: np.ndarray):        
        return np.maximum(0, x)
    def tanh(x : np.ndarray):
        return np.tanh(x)
    def sigmoid(x: np.ndarray):
        return 1 / (1 + np.exp(-x))
    def softmax(x: np.ndarray):
        x = np.exp(x-np.max(x, axis=-1, keepdims=True)) # Normalization term to avoid unbalanced output
        return x / np.sum(x, axis=-1, keepdims=True)

class Derivation:
    def relu(z: np.ndarray):        
        return z>0
    def tanh(x : np.ndarray):
        return 1 - np.square(np.tanh(x))
    def sigmoid(z: np.ndarray):
        return np.exp(-z) / np.square(1 + np.exp(-z))
    def softmax(z: np.ndarray):
        s = Activation.softmax(z)
        return s*(1-s)


class Layer:
    def __init__(self, input_shape = None, output_shape = None, name : str ="") -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        self.kernel = None
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        if len(input_data.shape) > 3: input_data = Flatten(input_shape=input_data.shape, end_dim=-1)(input_data)
        return input_data
    
    def rename(self, name : str):
        self.name = name
    
    def get_params_count(self):
        return 0
    
    def backward(self, dLoss_per_output : np.ndarray):
        return dLoss_per_output, None
    
    def scale_kernel(self, factor: float):
        return
    
class Flatten(Layer):
    def __init__(self, input_shape, start_dim=0, end_dim=None) -> None:
        super().__init__(input_shape)
        self.start_dim = start_dim
        self.end_dim = end_dim        
        self.output_shape = tuple()
        if self.start_dim != 0 : self.output_shape += self.input_shape[0 : self.start_dim]
        self.output_shape += (np.prod(self.input_shape[self.start_dim : self.end_dim]), )
        if self.end_dim: self.output_shape += self.input_shape[self.end_dim: None]
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        input_data = super().__call__(input_data)
        batch_size = input_data.shape[0]
        return input_data.reshape((batch_size, ) + self.output_shape)

'''
#Test Flatten
a = Flatten(input_shape=(28, 28))
b = np.random.rand(10,28,28)
c = a(b)
print(c.shape, c)
'''

class Dropout(Layer):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.retention_rate : float = 1-p
        self.bernoulli_mask : np.ndarray = None
        self.x = None
        
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        if training:
            self.x = np.copy(input_data)
            input_data = super().__call__(input_data)
            self.bernoulli_mask = np.random.rand(*input_data.shape) < self.retention_rate
            return (self.bernoulli_mask * input_data).reshape(self.x.shape)
        else:
            return input_data

    def backward(self, dLoss_per_output : np.ndarray):
        dLoss_per_output = dLoss_per_output * self.bernoulli_mask
        return dLoss_per_output, None
    
'''
# Test Dropout
d = Dropout(0.25)    
a = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
    ], dtype=object)
print(a.shape, d(a))
'''

class BatchNormalization(Layer):
    def __init__(self, in_features, epsilon:float=1e-7) -> None:
        super().__init__()
        self.z = None
        self.x = None
        self.x_hat = None
        self.epsilon = epsilon
        self.kernel = np.ones(shape=(in_features))
        self.biases = np.zeros(shape=(in_features))
        self.curr_esperance, self.curr_variance = None, None
        self.running_esperance, self.running_variance = 0, 0
        self.batch_size, self.batch_step = 0, 0
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        if training: 
            self.x = np.copy(input_data)
            input_data = super().__call__(input_data)
            self.batch_size = len(input_data)
            self.batch_step += 1
            self.curr_esperance, self.curr_variance = np.mean(input_data, axis=0), np.var(input_data, axis=0)
            self.running_esperance = self.running_esperance + self.curr_esperance
            self.running_variance = self.running_variance + self.curr_variance
            self.x_hat = (input_data - self.curr_esperance) / (np.sqrt(self.curr_variance + self.epsilon))            
        else:
            training_esperance = self.running_esperance / self.batch_step
            training_variance = (self.batch_size / (self.batch_size-1)) * (self.running_variance / self.batch_step) 
            self.x_hat = (input_data - training_esperance) / (np.sqrt(training_variance + self.epsilon))
        self.z = self.x_hat * self.kernel + self.biases
        return self.z
    
    def backward(self, dLoss_per_output) -> tuple[np.ndarray]:        
        dLoss_per_x_hat = dLoss_per_output * self.kernel
        
        dLoss_per_variance = (dLoss_per_x_hat * (self.x - self.curr_esperance) * \
            -(1/2) * (self.curr_variance + self.epsilon)**(-3/2)).sum(axis=0)

        dLoss_per_esperance = (dLoss_per_x_hat * -1 / np.sqrt(self.epsilon + \
            self.curr_variance)).sum(axis=0) + dLoss_per_variance * \
            -2 * np.mean(self.x - self.curr_esperance, axis=0)
        
        dLoss_per_x = (dLoss_per_x_hat * 1 / (np.sqrt(self.epsilon + self.curr_variance))) + \
            dLoss_per_variance * 2 * (self.x - self.curr_esperance) / len(self.x) + \
            dLoss_per_esperance * 1 / len(self.x)
                
        dLoss_per_gamma = np.sum(dLoss_per_output * self.x_hat, axis=0)
        dLoss_per_beta = np.sum(dLoss_per_output, axis=0)       
        return dLoss_per_x, (dLoss_per_gamma, dLoss_per_beta)
    
    def update_params(self, dGamma: np.ndarray, dBeta: np.ndarray):
        self.kernel += dGamma
        if self.biases is not None: self.biases += dBeta
'''
# Test Batch Norm
a=BatchNormalization(3)
ones = np.ones((10, 3))
input = np.random.randn(10, 3)
b=a(input)
print(b.shape)
_, dL = Loss()(b, input)
print("Before backward", dL)
dL, _ = a.backward(dL)
print("After backward", dL)
'''

class Dense(Layer):
    # -------------------------------------------------------------------#
    # Fully connected layer
    # -------------------------------------------------------------------#
    def __init__(self, in_features, out_features, activation = "reLu", bias=True, classes : list = None) -> None:
        # Input should be like (*, in_features) and output (*, out_features)
        super().__init__()
        assert not(activation == "softmax" and out_features < 2)
        assert not(activation == "softmax" and classes == None)
        self.activation = activation
        self.classes = classes
        normalization_term = 1
        if activation == "reLu":
            normalization_term = np.sqrt(2/in_features)
        if activation == "tanh":
            normalization_term = np.sqrt(1/in_features)
        else:
            normalization_term = np.sqrt(1 / (in_features+out_features))  
        self.kernel = np.random.randn(in_features, out_features)*normalization_term
        self.biases = np.random.randn(out_features) if bias else None
        self.z = None
        self.x = None
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        self.x = np.copy(input_data)
        input_data = super().__call__(input_data)       
        self.z = np.dot(input_data, self.kernel) 
        if self.biases is not None: self.z += self.biases
        return self.activate(self.z)
    
    def get_params_count(self):
        num_params = np.prod(self.kernel.shape) 
        if self.biases is not None: num_params += self.biases.shape[0]
        return num_params
    
    def get_classes(self):
        return self.classes
    
    def activate(self, z: np.ndarray) -> np.ndarray:
        # Output shape : z.shape
        match self.activation:
            case "identity":
                a = z 
            case "reLu":
                a = Activation.relu(z)
            case "sigmoid": 
                a = Activation.sigmoid(z)
            case "softmax":
                a = Activation.softmax(z)
        return a
        
    def derivate(self, z: np.ndarray) -> np.ndarray:
        # Output shape : z.shape
        match self.activation:
            case "identity":
                dOut_per_z = np.ones(z.shape)
            case "reLu":
                dOut_per_z = Derivation.relu(z)
            case "sigmoid":
                dOut_per_z = Derivation.sigmoid(z)
            case "softmax":
                dOut_per_z = Derivation.softmax(z)
        # print("derivate ", dOut_per_z)
        assert dOut_per_z.shape == z.shape
        # print("Layer: ", dOut_per_z)
        return dOut_per_z

    def backward(self, dLoss_per_output) -> tuple[np.ndarray]:
        dOut_per_z = self.derivate(self.z) # shape: z.shape
        dLoss_per_z = dLoss_per_output * dOut_per_z # shape: z.shape
        dLoss_per_W = np.dot(self.x.T, dLoss_per_z) # shape: W.shape
        dLoss_per_b = np.sum(dLoss_per_z, axis=0, keepdims=True) # shape: b.shape
        dLoss_per_output = np.dot(dLoss_per_output * dOut_per_z, self.kernel.T) # shape: x.shape
        return dLoss_per_output, (dLoss_per_W, dLoss_per_b)
        
    def update_params(self, dW: np.ndarray, db: np.ndarray, **kwargs):
        if kwargs:
            if 'clip_value' in kwargs: 
                dW = np.clip(dW, a_min=-kwargs['clip_value'], a_max=kwargs['clip_value'])            
                db = np.clip(db, a_min=-kwargs['clip_value'], a_max=kwargs['clip_value'])
        self.kernel += dW
        if self.biases is not None: self.biases += db.reshape(self.biases.shape)
    
    def scale_kernel(self, factor: float):
        self.kernel *= factor
'''
#Test Dense
a = Dense(input_shape=(1,2,3), out_features=2, activation="softmax")
b =a.process(np.random.randn(1,2, 3))
print(b.shape, b)
'''

class NeuralNet:
    def __init__(self, usage:Usage = Usage.regression) -> None:
        #self.input_shape = input_shape
        self.layers_stack : list[Layer] = []
        self.cache : list[dict[str, np.ndarray]] = [] # prediction - train_loss - val_loss - dL/dOut - gradients
        self.layers_counter : dict[str, any] = {} # layers counters for each type
        self.loss_fn : Loss = None
        self.regularizer : Regularizer = None
        self.optimizer : Optimizer = None
        self.dropout : float = None
        self.training : bool = True
        self.batch_size = None
        self.usage = usage
        self.training_steps_tracker = 0
    
    def add_layer(self, layer: Layer):
        layer_class = type(layer).__name__
        if not layer_class in self.layers_counter: 
            layer.rename(layer_class.lower())
            self.layers_counter[layer_class] = 1            
        else: 
            layer.rename(layer_class.lower() + '_' + str(self.layers_counter[layer_class]))
            self.layers_counter[layer_class] += 1
        self.layers_stack.append(layer)
    
    def pop_cache(self) -> dict[str, any]:
        return self.cache.pop()
    
    def init_cache(self) -> None:        
        self.cache.clear()
        zeros = [(0, 0) for l in self.layers_stack]        
        self.cache.append({'prediction': None, 'train_loss': None, 'val_loss': None, 
                           'dL/dOut': None, 'gradients': zeros})
        return
    
    def get_cache(self, key: str):
        if key not in self.cache[-1]: return None
        else: return np.array([data[key] for data in self.cache[1:]])
    
    def update_steps_tracker(self):
        self.training_steps_tracker += 1
        return
    
    def reset_steps_tracker(self):
        self.training_steps_tracker = 0
    
    def set_lr_scheduler(self, data_size : int, batch_size : int, num_epochs : int):
        steps_per_epoch = 1 if not batch_size else math.ceil(data_size/batch_size)
        self.optimizer.lr.set_steps_per_epoch(steps_per_epoch)
        self.optimizer.lr.set_decays_step(num_epochs*steps_per_epoch)
        return
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for layer in self.layers_stack:
            output = layer(output, self.training)
        if self.training: self.cache.append({'prediction': output})
        return output

    def compute_loss(self, y_recorded: np.ndarray) -> float:
        loss, dLoss_per_output = self.loss_fn(self.cache[-1]['prediction'], y_recorded)        
        if self.training:
            if self.regularizer: loss += self.regularizer.loss_reg(self.trainable_params())
            self.cache[-1]['train_loss'] = loss
            self.cache[-1]['dL/dOut'] = dLoss_per_output
        else: self.cache[-1]['val_loss'] = loss
        return loss
    
    def backpropagate(self):
        # -------------------------------------------------------------------#
        # Backpropagate the prediction error into all layers
        # -------------------------------------------------------------------#
        self.cache[-1]['gradients'] = []
        dLoss_per_output = self.cache[-1]['dL/dOut']
        for layer in reversed(self.layers_stack):
            dLoss_per_output, dLoss_per_params = layer.backward(dLoss_per_output)                
            self.cache[-1]['gradients'].append(dLoss_per_params)
        for l_index, layer in enumerate(reversed(self.layers_stack)):
            if self.cache[-1]['gradients'][l_index] is not None:
                # Optimization and regularization            
                if self.regularizer: 
                    self.cache[-1]['gradients'][l_index][0] += self.regularizer.gradients_reg(layer.kernel)
                if self.optimizer:
                    self.cache[-1]['gradients'][l_index] = self.optimizer(self.cache, l_index, step=self.training_steps_tracker) 
                gradient_W, gradient_biases = self.cache[-1]['gradients'][l_index]
                # Normalization
                gradient_W, gradient_biases = gradient_W/len(layer.z), gradient_biases/len(layer.z)
                self.cache[-1]['gradients'][l_index] = gradient_W, gradient_biases
                layer.update_params(-gradient_W, -gradient_biases)
                
    def disable_dropout(self, p: float):
        if not p: return
        for layer in self.layers_stack:
            if layer != self.layers_stack[-1]: layer.scale_kernel(p)
               
    def SGD(self, X: np.ndarray, Y: np.ndarray, validation_data : tuple[np.ndarray]):
        # Stochastic Gradient Descent
        self.training = True
        for i in (pepoch := tqdm(range(len(X)), colour='green')):
            x, y = np.expand_dims(X[i], axis=0), np.expand_dims(Y[i], axis=0)
            self.forward(x)
            loss = self.compute_loss(y)
            self.backpropagate()
            if i<(len(X)-1): self.pop_cache()
            else:
                eval_metrics = {'train_loss': loss}
                eval_metrics.update(self.evaluate(validation_data[0], validation_data[1], validation=True))
                pepoch.set_postfix(eval_metrics) # Progress bar infos
            self.update_steps_tracker()
            
    def BGD(self, X: np.ndarray, Y: np.ndarray, validation_data : tuple[np.ndarray]):
        # Batch Gradient Descent
        self.training = True
        self.forward(X)
        loss = self.compute_loss(Y)
        self.backpropagate()
        eval_metrics = {'train_loss': loss}
        eval_metrics.update(self.evaluate(validation_data[0], validation_data[1], validation=True))
        print(eval_metrics)
        self.update_steps_tracker()
    
    def MBGD(self, batch_size : int, X: np.ndarray, Y: np.ndarray, validation_data : tuple[np.ndarray]):
        # Mini Batch Gradient Descent
        self.training = True
        num_steps = (len(Y) // batch_size) + (len(Y) % batch_size > 0)
        for i in (pbatch := tqdm(range(num_steps), colour='green')):
            start, end = i*batch_size, min(len(X), (i+1)*batch_size)
            x, y = X[start:end], Y[start:end]
            self.forward(x)
            loss = self.compute_loss(y)
            self.backpropagate()
            if i<num_steps-1: self.pop_cache()
            else: 
                eval_metrics = {'train_loss': loss}
                eval_metrics.update(self.evaluate(validation_data[0], validation_data[1], validation=True))
                pbatch.set_postfix(eval_metrics) # Progress bar infos
            self.update_steps_tracker()
            
    def compile(self, learning_rate : float | LearningRateScheduler = 0.01, loss_fn: str | Loss = "l2", regularizer: Regularizer=None, 
                        dropout: float = None, optimizer: Optimizer=None):        
        self.loss_fn = Loss(loss_fn) if type(loss_fn) != Loss else loss_fn
        self.optimizer = (optimizer if optimizer else Optimizer(learning_rate))
        self.init_cache()
        self.regularizer = regularizer        
        self.dropout = dropout
        if dropout and (n_layers := len(self.layers_stack)) > 1:            
            [self.layers_stack.insert(i, Dropout(dropout)) for i in range(1, n_layers+1, 2)]
        return
    
    def train_one_epoch(self, train_X: np.ndarray, train_Y: np.ndarray, validation_data : tuple[np.ndarray],
                    batch_size: int=None):
        X, Y = train_X, train_Y
        # Way of training
        if not batch_size:
            self.BGD(X, Y, validation_data)
        elif batch_size==1:
            self.SGD(X, Y, validation_data)
        else:
            self.MBGD(batch_size, X, Y, validation_data)
        return
    
    def train(self, train_X: np.ndarray, train_Y: np.ndarray, nepochs=100, batch_size: int=None, 
              validation_data : tuple[np.ndarray] = None):
        self.training = True        
        self.reset_steps_tracker()
        self.batch_size = batch_size        
        if validation_data: val_X, val_Y = validation_data
        else: (train_X, train_Y), (val_X, val_Y) = Processing.train_test_split(train_X, train_Y, 0.2)
        train_X, train_Y = Processing.shuffle_dataset(train_X, train_Y)
        self.set_lr_scheduler(data_size=len(train_X), batch_size=batch_size, num_epochs=nepochs)
        print("Training data dimensions: ", train_X.shape)
        for e in range(1, nepochs+1):
            print(f"Epoch {str(e)}/{str(nepochs)}")
            self.train_one_epoch(train_X, train_Y, (val_X, val_Y), batch_size)                
        self.display_losses()

    def evaluate(self, X: np.ndarray, Y: np.ndarray, validation : bool = False) -> dict[str, any]:
        assert X.shape[0] == Y.shape[0]
        X, Y = Processing.shuffle_dataset(X, Y)
        self.training = False
        eval_metrics = {}
        if self.dropout: self.disable_dropout(self.dropout)
        
        if not self.batch_size or (len(Y) // self.batch_size) < 5:
            loss = self.loss_fn((Y_hat:=self.forward(X)), Y)[0]
        else:
            Y_hat = None
            loss = 0
            num_steps = (len(Y) // self.batch_size) + (len(Y) % self.batch_size > 0)
            for i in range(num_steps):
                start, end = i*self.batch_size, min((i+1)*self.batch_size, len(Y))
                x, y = X[start:end], Y[start:end]
                y_hat = self.forward(x)
                if Y_hat is None: Y_hat = y_hat
                else: Y_hat = np.vstack([Y_hat, y_hat])
                loss += self.loss_fn(y_hat, y)[0]
            loss /= (len(Y) // self.batch_size)     
        
        if self.dropout: self.disable_dropout(1/self.dropout)
        
        if validation:
            eval_metrics['val_loss'] = loss
            self.cache[-1]['val_loss'] = loss
            if self.usage != Usage.regression: 
                eval_metrics['accuracy'] = Accuracy(Y_hat, Y, usage=self.usage, verbose=False).output            
        else:
            eval_metrics['test_loss'] = loss
            if self.usage != Usage.regression: 
                eval_metrics['accuracy'] = Accuracy(Y_hat, Y, usage=self.usage, verbose=True).output
            if self.usage == Usage.multiClassification:                 
                eval_metrics['conf_mat'] = (ConfusionMatrix(Y_hat, Y, self.layers_stack[-1].get_classes(), 
                    usage=self.usage, verbose=True)).output
        return eval_metrics
    
    def cross_validate(self, k : int, train_X: np.ndarray, train_Y: np.ndarray, 
                                nepochs=100, batch_size: int=None):
        self.training = True
        self.batch_size = batch_size
        print("Input data dimensions: ", train_X.shape)
        kfolds = Processing.KFold(train_X, train_Y, k)
        global_train_losses = np.zeros((nepochs))
        global_val_losses = np.zeros((nepochs))
        for i, (train_fold, test_fold) in enumerate(kfolds):
            print(f"KFOLD {i+1}/{k}")
            model_copy = copy.deepcopy(self)
            model_copy.init_cache()
            for e in range(1, nepochs+1):
                print(f"Epoch {str(e)}/{str(nepochs)}")            
                model_copy.train_one_epoch(train_fold[0], train_fold[1], (test_fold[0], test_fold[1]), batch_size)
            global_train_losses += model_copy.get_cache('train_loss')
            global_val_losses += model_copy.get_cache('val_loss')
        global_train_losses /= k
        global_val_losses /= k
        self.display_losses(global_train_losses), self.display_losses(global_val_losses)
        return
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.training = False
        if self.dropout: self.disable_dropout(self.dropout)
        if not self.batch_size: 
            return(self.forward(X))
        else:
            Y_hat = None
            num_steps = (len(X) // self.batch_size) + (len(X) % self.batch_size > 0)
            for i in range(num_steps):
                start = i*self.batch_size
                end = min((i+1)*self.batch_size, len(X))
                y_hat = self.forward(X[start:end, :])
                if Y_hat is None: Y_hat = y_hat
                else: Y_hat = np.vstack([Y_hat, y_hat])
        if self.dropout: self.disable_dropout(1/self.dropout)    
        return Y_hat
        
    def trainable_params(self) -> np.ndarray:
        params = []
        for layer in self.layers_stack:
            params.append(layer.kernel)
        params = np.array(params, dtype=object)
        return params
    
    def summary(self):
        # function to describe the layers and parameters
        total = 0
        print("Model : 'sequential' ")
        print("________________________________________________________\n")
        print("Layer (type)\t\t\t Params #")
        print("========================================================\n")
        for layer in self.layers_stack:
            print(layer.name + " (" + type(layer).__name__ + ")" + "\t\t\t", end='')
            print(str(layer.get_params_count())+"\n")
            total += layer.get_params_count()
        print("========================================================\n")
        print("Total params: " + str(total) + "\n\n")
        
    def draw(self):
        fig, ax = plt.subplots()
        y = 0
        r = 0.5
        offset = 4
        x = r
        visual_nodes = [0]*self.total_nodes
        beautiful_colors = [(0.25,0.88,0.82),
                            (0.8,0.6,1),
                            (1,0.7,0.5),
                            (1,0.85,0.35),
                            (0.53,0.81,0.92),
                            (0.58,0.44,0.86),
                            (0.5,0.5,0),
                            (0.99,0.81,0.87)]

        for i in range(len(self.layers_stack)):
            # Draw the layer nodes
            layer = self.layers_stack[i]
            y = r
            #color = random.choice(beautiful_colors)
            color=(random.random(), random.random(),
                   random.random())
            bias = layer.nodes[-1]
            for node in layer.nodes:
                # Draw circles (nodes)                
                tmp = visual_nodes[node['id']] = Circle((x, y), r, fill=False, color=color )                
                ax.add_artist(tmp)
                ax.text(x=tmp.center[0], y=tmp.center[1], s=node['id'] if node != bias or i == len(self.layers_stack)-1 
                        else f"{node['id']} (bias)", color=color)
                y += r+1
            
            x += offset

        # Draw forward arrows
        for layer in self.layers_stack[:-1]:
            for node in layer.nodes:
                for connected_node in node['output_nodes']:
                    A = visual_nodes[node['id']].center[0]+r, visual_nodes[node['id']].center[1]
                    B = visual_nodes[connected_node['id']].center[0]-r, visual_nodes[connected_node['id']].center[1]

                    opposed_side_length = A[1] - B[1]
                    hypotenuse_length = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
                    assert hypotenuse_length >= 0
                    epsilon = 1e-10
                    hypotenuse_length = max(hypotenuse_length, epsilon)

                    angle = math.degrees(math.asin(opposed_side_length/hypotenuse_length))
                    ax.add_artist(ConnectionPatch(A, B, "data", "data", arrowstyle='->', color='blue'))
                    label= self.get_weight_value(self.get_weight(f"w{node['id']}{connected_node['id']}")) 
                    ax.text(x = (A[0]+B[0])/2, y=(A[1]+B[1])/2, 
                            s=f"{label:.2f} ", rotation=-angle, 
                            rotation_mode='anchor', ha='center', va="center")
                            
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        plt.title("Model")
        fig.tight_layout()
        plt.show()
        pass

    def save(self, model_name):
        # Save the model in the working directory
        with open(f"{model_name}.ssj","wb") as save_file:
            pickle.dump(self, save_file)
        
    def display_losses(self, *args, **kwargs):
        if not args and not kwargs:
            train_losses = self.get_cache('train_loss')
            val_losses = self.get_cache('val_loss')           
            nepochs = range(1, len(train_losses)+1)
            ncols = (train_losses is not None) + (val_losses is not None)
            axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10,5))[1]
            if train_losses is not None:
                axs[0].plot(nepochs, train_losses, "blue")
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('Training Loss')
            if val_losses is not None:
                axs[1].plot(nepochs, val_losses, "green")
                axs[1].set_xlabel('Epoch')
                axs[1].set_ylabel('Validation Loss')
        else:
            test_losses = args[0]
            nepochs = range(1, len(test_losses)+1)
            plt.plot(nepochs, test_losses, "blue")
        plt.show()
    
    def show_lr_evolution(self):
        self.optimizer.show_lr_evolution()


class Recurrent(Layer):
    def __init__(self, in_features : int, hidden_features : int, out_features : int, activation : str = "tanh") -> None:
        # Input should be like (in_features, batch_size)
        super().__init__()
        self.activation = activation
        self.x = None # shape (nx, m, T_x)
        self.a = None # shape (na, m, T_x)
        self.zy = None # shape (ny, m, T_y)
        self.Waa = np.random.randn(hidden_features, hidden_features)*np.sqrt(1/hidden_features) # shape (na, na)
        self.Wax = np.random.randn(hidden_features, in_features)*np.sqrt(1/hidden_features) # shape (na, nx)
        self.Wy =  np.random.randn(out_features, hidden_features)*np.sqrt(1/out_features)# shape (ny, na)
        self.ba = np.random.randn(hidden_features, 1) # shape (na, )
        self.by = np.random.randn(out_features, 1) # shape (ny, )
        self.cache = [{'a': None, 'y' : None}] # hidden states and predictions
        self.time_cache = [{'dWy': None, 'dby': None, 'dWaa': None, 'dWax': None, 'dba': None}]
        
    def forward_cell(self, input_data: np.ndarray, a_prev : np.ndarray, training : bool = True) -> np.ndarray:
        za_t = np.dot(self.Wax, input_data) + np.dot(self.Waa, a_prev) + self.ba 
        if self.activation == "tanh": a_next = Activation.tanh(za_t)
        elif self.activation == "reLu": a_next = Activation.relu(za_t)        
        zy_t = np.dot(self.Wy, a_next) + self.by        
        return a_next, zy_t

    def backward_cell(self, dLoss_per_output : np.ndarray, t : int) -> tuple[np.ndarray]:
        # First step compute gradients for output parameters
        dLoss_per_zy_t = dLoss_per_output * Derivation.softmax(self.zy[:, :, t])
        dLoss_per_by = np.sum(dLoss_per_zy_t, axis=0, keepdims=True)
        dLoss_per_Wy = np.dot(dLoss_per_output, self.a[:, :, t].T)
        dLoss_per_a_t = np.dot(self.Wy.T, dLoss_per_output)        
        # Now compute gradients for hidden state parameters
        dLoss_per_za_t = dLoss_per_a_t * (1 - np.square(self.a[:, :, t]))
        dLoss_per_ba = np.sum(dLoss_per_za_t, axis=0, keepdims=True)
        dLoss_per_Wax = np.dot(dLoss_per_za_t, self.x[:, :, t].T)
        
        if t >= 0 :
            dLoss_per_Waa = np.dot(dLoss_per_za_t, self.a[:, :, t-1].T)
            dLoss_per_a_prev = np.dot(self.Waa.T, dLoss_per_za_t)
        else: 
            dLoss_per_Waa, dLoss_per_a_prev = 0, 0

        self.time_cache.append(
            {'dWy': dLoss_per_Wy, 'dby': dLoss_per_by, 'dWaa': dLoss_per_Waa, 'dWax': dLoss_per_Wax, 'dba': dLoss_per_ba}
        )
        return dLoss_per_a_prev
    
    def forward_one_batch(self, input_data: np.ndarray) -> np.ndarray:
        # Input should be like (in_features, batch_size, seq)
        n_x, m, T_x = input_data.shape
        self.x = input_data
        self.a = np.zeros(shape=(self.ba.shape[0], m, T_x))
        self.zy = np.zeros(shape=(self.by.shape[0], m, T_x))
        prediction = np.zeros(shape=(self.by.shape[0], m, T_x))
        a_prev = self.a[:, :, 0]        
        for t in range(T_x):
            x_t = input_data[:, :, t]
            a_prev, zy_t = self.forward_cell(x_t, a_prev)
            self.a[:, :, t] = a_prev
            self.zy[:, :, t] = zy_t
            # self.cache.append({'a' : a_prev, 'y': y_t})
            prediction[:, :, t] = Activation.softmax(zy_t)
        return prediction
    
    def backward_one_batch(self, dLoss_per_output : np.ndarray):
        T_y = dLoss_per_output.shape[-1]
        for t in reversed(range(T_y)):
            print(t)
            dLoss_per_a_prev = self.backward_cell(dLoss_per_output[:, :, t], t)
        return

# Test Reccurent Layer
in_features=5
hidden_features=3
out_features=2
batch = 10
T = 5
a = Recurrent(in_features, hidden_features, out_features)
input = np.random.randn(in_features, batch, T)
output = np.random.randn(out_features, batch, T)
y = a.forward_one_batch(input)
_, L = Loss()(y, output)
a.backward_one_batch(L)
print(a.time_cache)


class LSTM(Layer):
    def __init__(self, in_features : int, hidden_features : int, out_features : int, activation : str = "tanh") -> None:
        # Input should be like (in_features, batch_size)
        super().__init__()
        self.activation = activation        
        self.x_t, self.y_t = None, None
        
        self.Wy =  np.random.randn(out_features, hidden_features)*np.sqrt(1/out_features) # prediction weights of shape (ny, na)
        self.by = np.random.randn(out_features, 1) # shape (ny, )
        
        # Gates and cells info
        self.Wf = np.random.randn(hidden_features, in_features + hidden_features)*np.sqrt(1/hidden_features) # forget gate weights
        self.bf = np.random.randn(hidden_features, 1) # forget gate bias
        self.ft = None # forget gate
        
        self.Wu = np.random.randn(hidden_features, in_features + hidden_features)*np.sqrt(1/hidden_features) # update gate weights
        self.bu = np.random.randn(hidden_features, 1) # update gate bias
        self.ut = None # update gate
        
        self.Wo = np.random.randn(hidden_features, in_features + hidden_features)*np.sqrt(1/hidden_features) # output gate weights
        self.bo = np.random.randn(hidden_features, 1) # output gate bias
        self.ot = None # output gate
        
        self.Wc = np.random.randn(hidden_features, in_features + hidden_features)*np.sqrt(1/hidden_features) # candidate state weights
        self.bc = np.random.randn(hidden_features, 1) # candidate state bias
        
        self.c_t = None # cell state
        self.cct = None # candidate cell state
        self.a_t = None # hidden state
        
        self.cache = [{'a': None, 'c': None,'y' : None}] # hidden states and predictions
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        self.x_t = np.copy(input_data)
        input_data = super().__call__(input_data)
        concat = np.concatenate(input_data, self.a_t)
        
        self.ft = Activation.sigmoid(np.dot(self.Wf, concat) + self.bf)
        self.ut = Activation.sigmoid(np.dot(self.Wu, concat) + self.bu)
        self.cct = Activation.tanh(np.dot(self.Wc, concat) + self.bc)
        self.c_t = self.ft * self.c_t + self.ut * self.cct
        self.ot = Activation.sigmoid(np.dot(self.Wo, concat) + self.bo)
        self.a_t = self.ot * Activation.tanh(self.c_t)
        self.y_t = Activation.softmax(np.dot(self.Wy, self.a_t) + self.by)
        
        self.cache.append({'a': self.a_t, 'c': self.c_t, 'y' : self.y_t})
        return 
    
    def forward_one_epoch(self, input_data: np.ndarray) -> np.ndarray:
        # Input should be like (in_features, batch_size, seq)
        n_x, m, T_x = input_data.shape
        if self.a_t is None : self.a_t = np.zeros(shape=(self.bc.shape[0], m))
        if self.c_t is None : self.c_t = np.zeros(shape=(self.bc.shape[0], m))     
        for t in range(T_x):
            x_t = input_data[:, :, t]
            a_prev, _ = self.__call__(x_t)
            
class RNN :
    def __init__(self, in_features : int, hidden_features : int, num_layers : int = 1, activation : str = "tanh") -> None:
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.activation = activation
        self.cells = []
        
        self.cache : list[dict[str, np.ndarray]] = [] # prediction - train_loss - val_loss - dL/dOut - gradients
        self.loss_fn : Loss = None
        self.regularizer : Regularizer = None
        self.optimizer : Optimizer = None
        self.dropout : float = None
        self.training : bool = True
        self.batch_size = None
        # self.usage = usage
        self.training_steps_tracker = 0
        
    def add_cells(self, Tx : int = 2):
        self.cells = [Recurrent(self.in_features, self.hidden_features, self.num_layers, self.activation) for _ in range(Tx)]
        return len(self.cells)
    
    def pop_cache(self) -> dict[str, any]:
        return self.cache.pop()
    
    def init_cache(self) -> None:        
        self.cache.clear()
        self.cache.append({'prediction': None, 'train_loss': None, 'val_loss': None, 
                           'dL/dOut': None, 'gradients': 0})
        return
    
    def get_cache(self, key: str):
        if key not in self.cache[-1]: return None
        else: return np.array([data[key] for data in self.cache[1:]])

    def update_steps_tracker(self):
        self.training_steps_tracker += 1
        return
    
    def reset_steps_tracker(self):
        self.training_steps_tracker = 0
    
    def set_lr_scheduler(self, data_size : int, batch_size : int, num_epochs : int):
        steps_per_epoch = 1 if not batch_size else math.ceil(data_size/batch_size)
        self.optimizer.lr.set_steps_per_epoch(steps_per_epoch)
        self.optimizer.lr.set_decays_step(num_epochs*steps_per_epoch)
        return
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # Input should be like (in_features, batch_size, seq)
        n_x, m, T_x = input_data.shape
        self.add_cells(T_x)
        a_prev = None
        for t in range(T_x):
            x_t = input_data[:, :, t]
            cell_t = self.cells[t]
            a_prev = cell_t(x_t, a_prev)            
        
    
    def compute_loss(self, y_recorded: np.ndarray) -> float:
        return

def load(model_name):
    with open(f"{model_name}.ssj","rb") as save_file:
        load = pickle.load(save_file)
        return load


# if __name__ == '__main__':