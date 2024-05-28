import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sn
from tqdm import tqdm
from enum import Enum
import copy

def identity(input : np.ndarray):
    return input
def relu(input: np.ndarray):        
    return np.maximum(0, input)
def tanh(input : np.ndarray):
    return np.tanh(input)
def sigmoid(input: np.ndarray):
    return 1 / (1 + np.exp(-input))
def softmax(input: np.ndarray):
    input = np.exp(input-np.max(input, axis=-1, keepdims=True)) # Normalization term to avoid unbalanced output
    return input / np.sum(input, axis=-1, keepdims=True)

def didentity(activation: np.ndarray):
    return np.ones(activation.shape)
def drelu(activation: np.ndarray):        
    return activation>0
def dtanh(activation : np.ndarray):
    return 1 - np.square(activation)
def dsigmoid(activation: np.ndarray):
    return activation*(1-activation)
def dsoftmax(activation: np.ndarray):
    return activation*(1-activation)

def orthogonal_initializer(shape, gain=1.0):
    """
    Orthogonal initialization for the recurrent weights.
    """
    if len(shape) < 2:
        raise ValueError("The tensor to initialize must be at least two-dimensional")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q

def glorot_uniform_initializer(shape):
    """
    Glorot Uniform initializer.
    """
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def glorot_normal_initializer(shape):
    """
    Initialize a weight matrix using Glorot Normal (Xavier Normal) initialization.
    """
    fan_in, fan_out = shape[0], shape[1]
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(loc=0.0, scale=stddev, size=shape)

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

class Activation(Enum):
    identity = "identity"
    relu = "reLu",
    sigmoid = "sigmoid",
    tanh = "tanh",
    softmax = "softmax"
    
class RnnArchitecture(Enum):
    one_to_one = "one_to_one"
    many_to_one = "many_to_one"
    one_to_many = "one_to_many"
    many_to_many = "many_to_many"

class Processing():
    def __init__(self) -> None:
        pass
    
    def one_hot_encode_text(corpus : list , vocab_size : int):
        words_to_index = {}
        indexes = [[] for _ in range(len(corpus))]
        tokenized_corpus = [str(sentence).split(sep=' ') for sentence in corpus]
        for sentence in tokenized_corpus:    
            for word in sentence:
                if word not in words_to_index:
                    words_to_index[word] = random.randint(1, vocab_size-1)                    
        for i, sentence in enumerate(tokenized_corpus):
            for word in sentence:
                indexes[i].append(words_to_index[word])
        return indexes
    
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
        if usage == Usage.logisticRegression: 
            predictions = np.rint(predictions).astype(int)
            labels = np.rint(labels).astype(int)
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
        # "positive|negative" + "true|false"
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
    
    def loss_reg(self, all_kernels: np.ndarray) -> float:
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
    
    def gradients_reg(self, kernel: np.ndarray) -> np.ndarray:
        match self.name:
            case 'ridge':
                self.gradient_regularization = 2*self.alpha*kernel
            case 'lasso':
                self.gradient_regularization = self.alpha*np.sign(kernel)
            case 'elastic-net':
                self.gradient_regularization = self.beta*((1-self.alpha)*kernel + self.alpha*np.sign(kernel))
        return self.gradient_regularization


class Loss:
    def __init__(self, loss_function : str ="l2", epsilon : float = 1e-12) -> None:
        self.loss_fn = loss_function
        self.epsilon = epsilon
        self.error = None
        self.dLoss_per_prediction = None
    
    def __call__(self, y_predicted: np.ndarray, y_recorded: np.ndarray) -> tuple[float, np.ndarray]:
        # y_predicted and y_recorded should have shape (batch, *)
        batch_size = len(y_predicted)
        match self.loss_fn:
            case "l0-1":
                self.dLoss_per_prediction = 0
                self.error =  np.where(y_recorded != y_predicted, 1, 0).sum(axis=tuple(range(1, y_predicted.ndim)))
            case "hinge":
                self.dLoss_per_prediction =  np.where(1-y_recorded*y_predicted >=0, -y_recorded, 0)
                self.error = np.maximum(0, 1-y_recorded*y_predicted).sum(axis=tuple(range(1, y_predicted.ndim)))            
            case "l2" | "quadratic":
                self.dLoss_per_prediction = 2*(y_predicted-y_recorded)
                self.error = np.square(y_predicted-y_recorded).sum(axis=tuple(range(1, y_predicted.ndim)))
            case "l1":
                self.dLoss_per_prediction = np.sign(y_predicted)
                self.error = np.absolute(y_recorded-y_predicted).sum(axis=tuple(range(1, y_predicted.ndim)))
            case "binary_cross_entropy":
                assert len(y_recorded.shape) >= 2 and y_recorded.shape[-1] == 1
                # Clipping y_predicted to avoid log(0)
                y_predicted = np.clip(y_predicted, self.epsilon, 1 - self.epsilon)
                self.dLoss_per_prediction = (y_predicted-y_recorded) / (y_predicted * (1- y_predicted))
                self.error = - (y_recorded * np.log(y_predicted) + (1 - y_recorded) * np.log(1 - y_predicted)).sum(axis=tuple(range(1, y_predicted.ndim)))               
            case "categorical_cross_entropy":
                assert len(y_recorded.shape) >= 2 and y_recorded.shape[-1] > 1
                assert np.min(y_predicted) >= 0 and np.max(y_predicted) <= 1
                y_predicted = np.clip(y_predicted, self.epsilon, 1 - self.epsilon)
                self.dLoss_per_prediction = (y_predicted - y_recorded) # / y_recorded.shape[-1]
                self.error = (-y_recorded * np.log(y_predicted)).sum(axis=tuple(range(1, y_predicted.ndim)))
        self.error = np.mean(self.error)
        self.dLoss_per_prediction /= batch_size
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
    def __init__(self, lr : float | LearningRateScheduler=0.01, epsilon:float=1e-12) -> None:
        if not isinstance(lr, LearningRateScheduler): lr = LearningRateScheduler(lr)
        self.lr = lr
        self.epsilon = epsilon
        
    def __call__(self, gradients: np.ndarray, num_layer: int, step : int, epoch : int = None)->np.ndarray:
        return [self.lr(step) * gradient for gradient in gradients]

    def show_lr_evolution(self):
        self.lr.evolution()
        
class Momentum(Optimizer):
    def __init__(self, alpha: float=0.9, lr: float | LearningRateScheduler=0.01) -> None:
        super().__init__(lr)
        self.alpha = alpha
        self.previous_updates = {}
        
    def __call__(self, gradients: np.ndarray, num_layer: int, step : int, epoch : int = None):        
        if num_layer not in self.previous_updates:
            self.previous_updates[num_layer] = [np.zeros_like(gradient) for gradient in gradients]
        previous_update = self.previous_updates[num_layer]
        optim_gradients = [self.lr(step) * gradients[i]  + self.alpha * previous_update[i] for i in range(len(gradients))]
        self.previous_updates[num_layer] = optim_gradients
        return optim_gradients
    
class Adam(Optimizer):
    def __init__(self, beta_1: float = 0.9, beta_2: float=0.999, lr : float| LearningRateScheduler = 0.01, epsilon:float=1e-12) -> None:
        super().__init__(lr, epsilon)
        self.beta_1, self.beta_2 = beta_1, beta_2
        self.square_sums = {}
        self.linear_sums = {}
        
    def __call__(self, gradients: np.ndarray, num_layer: int, step : int, epoch : int):
        if num_layer not in self.square_sums:
            self.square_sums[num_layer] = [np.zeros_like(gradient) for gradient in gradients]
        if num_layer not in self.linear_sums:
            self.linear_sums[num_layer] = [np.zeros_like(gradient) for gradient in gradients]
        
        self.square_sums[num_layer] = [self.beta_2 * self.square_sums[num_layer][i] + 
                                       (1-self.beta_2)*np.square(gradients[i]) for i in range(len(gradients))]
        
        self.linear_sums[num_layer] = [self.beta_1*self.linear_sums[num_layer][i] + 
                                       (1-self.beta_1)*gradients[i] for i in range(len(gradients))]
        
        square_sum, linear_sum = self.square_sums[num_layer], self.linear_sums[num_layer]
        
        first_momentum = [(linear_sum[i] / (1 - self.beta_1**(step+1))) for i in range(len(gradients))]
        
        second_momentum = [(square_sum[i] / (1 - self.beta_2**(step+1))) for i in range(len(gradients))]
        
        optim_gradients = [self.lr(step) * first_momentum[i] / (self.epsilon + np.sqrt(second_momentum[i])) for i in range(len(gradients))]
        return optim_gradients
    
class Adagrad(Optimizer):
    def __init__(self, lr: float=0.01, epsilon:float=1e-12) -> None:
        super().__init__(lr, epsilon)
        self.square_sums = {}
        
    def __call__(self, gradients: np.ndarray, num_layer: int, step : int, epoch : int = None):
        if num_layer not in self.square_sums:
            self.square_sums[num_layer] = [np.zeros_like(gradient) for gradient in gradients]
        self.square_sums[num_layer] = [self.square_sums[num_layer][i] + np.square(gradients[i]) for i in range(len(gradients))]
        square_sum = self.square_sums[num_layer]
        optim_gradients = [gradients[i]*self.lr(step) / (self.epsilon + np.sqrt(square_sum[i])) for i in range(len(gradients))]
        return optim_gradients
    
class RMSprop(Optimizer):
    def __init__(self, beta: float=0.9, lr: float=0.01, epsilon:float=1e-12) -> None:
        super().__init__(lr, epsilon)
        self.beta = beta
        self.square_sums = {}

    def __call__(self, gradients: np.ndarray, num_layer: int, step : int, epoch : int = None):
        if num_layer not in self.square_sums:
            self.square_sums[num_layer] = [np.zeros_like(gradient) for gradient in gradients]
        self.square_sums[num_layer] = [self.beta*self.square_sums[num_layer][i] + 
                                       (1-self.beta)*np.square(gradients[i]) for i in range(len(gradients))]
        square_sum = self.square_sums[num_layer]
        optim_gradients = [gradients[i]*self.lr(step)/(self.epsilon + np.sqrt(square_sum[i])) for i in range(len(gradients))]
        return optim_gradients

   
class Layer:
    def __init__(self, input_shape = None, output_shape = None, name : str ="") -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        self.kernel = None
        self.freeze = None

    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        if len(input_data.shape) > 3: input_data = Flatten(input_shape=input_data.shape, end_dim=-1)(input_data)
        return input_data
    
    def rename(self, name : str):
        self.name = name
    
    def get_params_count(self):
        return 0
    
    def get_output(self):
        return None
    
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
    
    def get_output(self):
        return self.z
    
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
        return dLoss_per_x, [dLoss_per_gamma, dLoss_per_beta]
    
    def update_params(self, gradient : np.ndarray):
        dGamma, dBeta = gradient
        self.kernel += dGamma
        if self.biases is not None: self.biases += dBeta
    
    def get_params_count(self):
        num_params = np.prod(self.kernel.shape) 
        if self.biases is not None: num_params += self.biases.shape[0]
        return num_params
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
    def __init__(self, in_features, out_features, activation : str | Activation = "reLu", 
                 bias=True, classes : list = None, **kwargs) -> None:
        # Input should be like (*, in_features) and output (*, out_features)
        super().__init__()
        if not isinstance(activation, Activation):
            if activation in Activation._member_map_: activation = Activation._member_map_[activation]
            else: activation = Activation.relu
        assert not(activation == Activation.softmax and out_features < 2)
        assert not(activation == Activation.softmax and classes == None)
        self.activation = activation
        self.classes = classes
        normalization_term = 1
        if activation == Activation.relu:
            normalization_term = np.sqrt(2/in_features)
        if activation == Activation.tanh or activation == Activation.sigmoid:
            normalization_term = np.sqrt(1/in_features)
        else:
            normalization_term = np.sqrt(2 / (in_features+out_features))  
        self.kernel = np.random.randn(in_features, out_features)*normalization_term
        self.biases = np.random.randn(out_features) if bias else None
        if kwargs:
            if 'load_weights' in kwargs:
                self.kernel, self.biases = kwargs['load_weights']
            if 'freeze' in kwargs:
                self.freeze = kwargs['freeze']
        self.z = None
        self.x = None
    
    def get_output(self):
        return self.activate(self.z)
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        self.x = np.copy(input_data)
        input_data = super().__call__(input_data)       
        self.z = np.dot(input_data, self.kernel) 
        if self.biases is not None: self.z += self.biases
        self.z = self.activate(self.z)
        return self.z
    
    def get_params_count(self):
        num_params = np.prod(self.kernel.shape) 
        if self.biases is not None: num_params += self.biases.shape[0]
        return num_params
    
    def get_classes(self):
        return self.classes
    
    def activate(self, z: np.ndarray) -> np.ndarray:
        activation_function = globals()[self.activation.name]
        return activation_function(z)
        
    def derivate(self, z: np.ndarray) -> np.ndarray:
        derivation_function = globals()["d"+self.activation.name]
        return derivation_function(z)

    def backward(self, dLoss_per_output) -> tuple[np.ndarray]:
        dOut_per_z = self.derivate(self.z) # shape: z.shape
        dLoss_per_z = dLoss_per_output * dOut_per_z # shape: z.shape
        dLoss_per_W = np.dot(self.x.T, dLoss_per_z) # shape: W.shape
        dLoss_per_b = np.sum(dLoss_per_z, axis=0) # shape: b.shape
        dLoss_per_output = np.dot(dLoss_per_output * dOut_per_z, self.kernel.T) # shape: x.shape
        return dLoss_per_output, [dLoss_per_W, dLoss_per_b]
        
    def update_params(self, gradient : np.ndarray, **kwargs):
        dW, db = gradient
        if kwargs:
            if 'clip_value' in kwargs: 
                dW = np.clip(dW, a_min=-kwargs['clip_value'], a_max=kwargs['clip_value'])            
                db = np.clip(db, a_min=-kwargs['clip_value'], a_max=kwargs['clip_value'])
        if not self.freeze:
            self.kernel -= dW
            if self.biases is not None: self.biases -= db.reshape(self.biases.shape)
    
    def scale_kernel(self, factor: float):
        self.kernel *= factor
'''
#Test Dense
a = Dense(in_features=3, out_features=2, activation="sigmoid")
b = a(np.random.randn(2, 3))
c = a.derivate(b)
print(b.shape, b)
print(c.shape, c)
'''

class Embedding(Layer):
    def __init__(self, input_dim : int, output_dim : int, seq_length : int, **kwargs) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.kernel = np.random.uniform(low=-0.05, high=0.05, size=(input_dim, output_dim))
        if kwargs:
            if 'load_weights' in kwargs:
                self.kernel = kwargs['load_weights']
            if 'freeze' in kwargs:
                self.freeze = kwargs['freeze']
        self.x = None
        self.z = None

    def get_output(self):
        return self.z
    
    def __call__(self, input_data: np.ndarray[int], training : bool = True) -> np.ndarray:
        # input_data should be of shape (batch_size, sequence_length)
        self.x = input_data
        self.z = self.kernel[input_data]
        return self.z
    
    def backward(self, dLoss_per_output) -> tuple[np.ndarray]:
        # dLoss_per_output of shape (batch_size, sequence_length, out_dim)
        dLoss_per_W = np.zeros_like(self.kernel)
        # Accumulate gradients for each word index 
        np.add.at(dLoss_per_W, self.x, dLoss_per_output)
        dLoss_per_output = dLoss_per_output.sum(axis=-1)
        return None, [dLoss_per_W]
    
    def update_params(self, gradient: np.ndarray):
        # print("Gradient embedding : ", sum([np.sum(g) for g in gradient]))
        if not self.freeze:
            self.kernel += gradient[0]
        
    def get_params_count(self):
        return np.prod(self.kernel.shape) 

class RNN(Layer):
    def __init__(self, in_features : int, hidden_features : int, architecture : str | RnnArchitecture = "many_to_one", 
                 activation : str | Activation= "tanh", **kwargs) -> None:
        super().__init__()
        if not isinstance(activation, Activation):
            if activation in Activation._member_map_: activation = Activation._member_map_[activation]
            else: activation = Activation.tanh
        self.activation = activation
        if not isinstance(architecture, RnnArchitecture):
            if architecture in RnnArchitecture._member_map_: architecture = RnnArchitecture._member_map_[architecture]
            else: architecture = RnnArchitecture.many_to_one
        self.n_i, self.n_a, self.T_x = in_features, hidden_features, 0
        self.architecture = architecture
        self.x = None # shape (m, Tx, nx)
        self.a = None # shape (m, Tx, na)
        self.Waa = orthogonal_initializer(shape=(hidden_features, hidden_features)) # shape (na, na)
        self.Wxa = glorot_normal_initializer(shape=(in_features, hidden_features)) # shape (nx, na)
        self.ba = np.zeros((hidden_features, )) # shape (na, )
        if kwargs:
            if 'load_weights' in kwargs:
                self.Wxa, self.Waa, self.ba = kwargs['load_weights']
            if 'freeze' in kwargs:
                self.freeze = kwargs['freeze']
    
    def get_params_count(self):
        num_params = np.prod(self.Waa.shape) + np.prod(self.Wxa.shape) + len(self.ba)
        return num_params
    
    def get_output(self):
        return self.a
    
    def forward_cell(self, input_data_t: np.ndarray, a_prev : np.ndarray) -> np.ndarray:
        # input_data shape : (m, nx) - a_prev shape : (m, na)
        za_t = np.dot(input_data_t, self.Wxa) + np.dot(a_prev, self.Waa) + self.ba
        activation_function = globals()[self.activation.name]       
        a_next = activation_function(za_t)
        return a_next

    def backward_cell(self, dLoss_per_a_next : np.ndarray, t : int) -> tuple[np.ndarray]:
        derivation_function = globals()["d"+self.activation.name]
        dLoss_per_za_next = dLoss_per_a_next * derivation_function(self.a[:, t])
        dLoss_per_ba = np.sum(dLoss_per_za_next, axis=0)
        dLoss_per_Wxa = np.dot(self.x[:, t].T, dLoss_per_za_next)
        dLoss_per_x_t = np.dot(dLoss_per_za_next, self.Wxa.T)
        dLoss_per_a_prev = np.dot(dLoss_per_za_next, self.Waa.T)
        dLoss_per_Waa = np.dot(self.a[:, t-1].T, dLoss_per_za_next)
        self.time_cache['dWaa'] += dLoss_per_Waa
        self.time_cache['dWxa'] += dLoss_per_Wxa
        self.time_cache['dba'] += dLoss_per_ba
        return dLoss_per_a_prev, dLoss_per_x_t
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        # Input should be like (batch_size, seq, in_features)
        m, self.T_x, nx = input_data.shape
        self.x = input_data
        self.a = np.zeros(shape=(m, self.T_x, self.ba.shape[0]))
        a_prev = np.zeros((m, self.ba.shape[0]))       
        for t in range(self.T_x):
            x_t = input_data[:, t]
            a_prev = self.forward_cell(x_t, a_prev)
            self.a[:, t] = a_prev
        
        if self.architecture == RnnArchitecture.many_to_many:
            return self.a
        if self.architecture == RnnArchitecture.many_to_one:
            return a_prev
    
    def backward(self, dLoss_per_a : np.ndarray):
        # dLoss_per_a shape : (m, Tx, na)
        self.time_cache = {'dWaa': 0, 'dWxa': 0, 'dba': 0}
        dLoss_per_x = np.zeros_like(self.x)
        dLoss_per_a_prev = dLoss_per_a
        for t in reversed(range(self.T_x)):
            if self.architecture == RnnArchitecture.many_to_many:
                dLoss_per_a_prev, dLoss_per_x_t = self.backward_cell(dLoss_per_a[:, t], t)
            elif self.architecture == RnnArchitecture.many_to_one:
                dLoss_per_a_prev, dLoss_per_x_t = self.backward_cell(dLoss_per_a_prev, t)
            dLoss_per_x[:, t] = dLoss_per_x_t
        cumul_gradients = [gradient for gradient in self.time_cache.values()]
        return dLoss_per_x, cumul_gradients
    
    def update_params(self, gradient : np.ndarray):
        # print("Gradient RNN : ", sum([np.sum(g) for g in gradient]))
        if not self.freeze:
            self.Waa += gradient[0]
            self.Wxa += gradient[1]
            self.ba += gradient[2]
        return

'''
# Test Reccurent Layer
in_features=5
hidden_features=3
batch = 10
T = 5
a = RNN(in_features, hidden_features)
input = np.random.randn(batch, T, in_features)
y = a(input)
print(y.shape)
'''

class LSTM(Layer):
    def __init__(self, in_features : int, hidden_features : int, architecture : str | RnnArchitecture = "many_to_one", 
                 activation : str | Activation= "tanh") -> None:
        # Input should be like (in_features, batch_size)
        super().__init__()
        if not isinstance(activation, Activation):
            if activation in Activation._member_map_: activation = Activation._member_map_[activation]
            else: activation = Activation.tanh
        self.activation = activation
        if not isinstance(architecture, RnnArchitecture):
            if architecture in RnnArchitecture._member_map_: architecture = RnnArchitecture._member_map_[architecture]
            else: architecture = RnnArchitecture.many_to_one
        self.architecture = architecture
        
        self.n_i, self.n_a, self.T_x = in_features, hidden_features, 0
        # Gates and cells info
        self.Wf = glorot_uniform_initializer(shape=(in_features + hidden_features, hidden_features)) # forget gate weights
        self.bf = np.zeros(shape=(hidden_features,)) # forget gate bias
        self.ft = None # forget gate shape (m, Tx, na)
        
        self.Wu = glorot_uniform_initializer(shape=(in_features + hidden_features, hidden_features)) # update gate weights
        self.bu = np.zeros(shape=(hidden_features,)) # update gate bias
        self.ut = None # update gate shape (m, Tx, na)
        
        self.Wo = glorot_uniform_initializer(shape=(in_features + hidden_features, hidden_features)) # output gate weights
        self.bo = np.zeros(shape=(hidden_features,)) # output gate bias
        self.ot = None # output gate shape (m, Tx, na)
        
        self.Wc = glorot_uniform_initializer(shape=(in_features + hidden_features, hidden_features)) # candidate state weights
        self.bc = np.zeros(shape=(hidden_features,)) # candidate state bias
        
        self.x = None # shape (m, Tx, nx)
        self.c = None # cell state shape (m, Tx, na)
        self.candidate = None # candidate cell state shape (m, Tx, na)
        self.a = None # hidden state shape (m, Tx, na)
        
        self.time_cache = {'dWf': [], 'dWu': [], 'dWo': [], 'dWc' : [], 'bf' : [], 'bu': [], 'bo': [], 'bc': []}
    
    def get_params_count(self):
        num_params = np.prod(self.Wf.shape) + np.prod(self.Wu.shape) + np.prod(self.Wo.shape) + np.prod(self.Wc.shape) + \
            len(self.bf) + len(self.bu) + len(self.bo) + len(self.bc)
        return num_params

    def get_output(self):
        return self.a
    
    def forward_cell(self, input_data_t: np.ndarray, a_prev : np.ndarray, c_prev : np.ndarray) -> np.ndarray:
        # input_data shape : (m, nx) - a_prev shape : (m, na) - c_prev shape : (m, na)
        # print("Shapes", input_data_t.shape, a_prev.shape)
        concat = np.concatenate((a_prev, input_data_t), axis=-1)
        ft = sigmoid(np.dot(concat, self.Wf) + self.bf)
        ut = sigmoid(np.dot(concat, self.Wu) + self.bu)
        candidate = tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = ft * c_prev + ut * candidate
        ot = sigmoid(np.dot(concat, self.Wo) + self.bo)
        activation_function = globals()[self.activation.name]       
        a_next = ot * activation_function(c_next) 
        return a_next, c_next, candidate, ft, ut, ot
    
    def backward_cell(self, dLoss_per_a_t : np.ndarray, dLoss_per_c_t : np.ndarray, t : int) -> tuple[np.ndarray]:
        xt = self.x[:, t]
        concat = np.concatenate((self.a[:, t], xt), axis=-1)
        
        activation_function = globals()[self.activation.name]
        derivation_function = globals()["d"+self.activation.name]
        dLoss_per_c_t += dLoss_per_a_t * self.ot[:, t] * derivation_function(activation_function(self.c[:, t]))
        
        dLoss_per_ot = dLoss_per_a_t * activation_function(self.c[:, t])  * dsigmoid(self.ot[:, t])
        
        dLoss_per_candidate = dtanh(self.candidate[:, t]) * (dLoss_per_c_t * self.ut[:, t] \
            + self.ot[:, t] * derivation_function(self.c[:, t]) * self.ut[:, t] * dLoss_per_a_t)
        
        dLoss_per_ut = dsigmoid(self.ut[:, t]) * (dLoss_per_c_t * self.candidate[:, t] \
            + self.ot[:, t] * derivation_function(self.c[:, t]) * self.candidate[:, t] * dLoss_per_a_t)
        
        dLoss_per_ft = dsigmoid(self.ft[:, t]) * (dLoss_per_c_t * self.c[:, t-1] \
            + self.ot[:, t] * derivation_function(self.c[:, t]) * self.c[:, t-1] * dLoss_per_a_t)
            
        dLoss_per_Wo = np.dot(concat.T, dLoss_per_ot)
        dLoss_per_Wu = np.dot(concat.T, dLoss_per_ut)
        dLoss_per_Wf = np.dot(concat.T, dLoss_per_ft)
        dLoss_per_Wc = np.dot(concat.T, dLoss_per_candidate)
        
        dLoss_per_bo = np.sum(dLoss_per_ot, axis=0)
        dLoss_per_bu = np.sum(dLoss_per_ut, axis=0)
        dLoss_per_bf = np.sum(dLoss_per_ft, axis=0)
        dLoss_per_bc = np.sum(dLoss_per_candidate, axis=0)
            
        self.time_cache['dWf'].append(dLoss_per_Wf)
        self.time_cache['dWu'].append(dLoss_per_Wu)
        self.time_cache['dWo'].append(dLoss_per_Wo)
        self.time_cache['dWc'].append(dLoss_per_Wc)
        self.time_cache['dbf'].append(dLoss_per_bf)
        self.time_cache['dbu'].append(dLoss_per_bu)
        self.time_cache['dbo'].append(dLoss_per_bo)
        self.time_cache['dbc'].append(dLoss_per_bc)
        
        # Compute derivatives w.r.t previous hidden state, previous memory state and input.
        dLoss_per_concat = np.dot(dLoss_per_ft, self.Wf.T) + np.dot(dLoss_per_ot, self.Wo.T) + \
            np.dot(dLoss_per_ut, self.Wu.T) + np.dot(dLoss_per_candidate, self.Wc.T)
        dLoss_per_x_t = dLoss_per_concat[:, : self.n_i]
        dLoss_per_a_prev = dLoss_per_concat[:, self.n_i:]
        dLoss_per_c_prev = dLoss_per_c_t * self.ft[:, t] \
            + self.ot[:, t] *  derivation_function(self.c[:, t]) * self.ft[:, t] * dLoss_per_a_t
        
        return dLoss_per_x_t, dLoss_per_a_prev, dLoss_per_c_prev
    
    def __call__(self, input_data: np.ndarray, training : bool = True) -> np.ndarray:
        # Input should be like (batch_size, seq, in_features)
        m, self.T_x, nx = input_data.shape
        na = self.bf.shape[0]
        self.x = input_data
        self.a = np.random.randn(m, self.T_x, na)
        self.c = np.random.randn(m, self.T_x, na) # np.zeros(shape=(m, Tx, na))
        self.ft = np.zeros(shape=(m, self.T_x, na))
        self.ut = np.zeros(shape=(m, self.T_x, na))
        self.ot = np.zeros(shape=(m, self.T_x, na))
        self.candidate = np.zeros(shape=(m, self.T_x, na))
        
        a_prev = np.zeros(shape=(m, na))
        c_prev = np.zeros(shape=(m, na))

        for t in range(self.T_x):
            x_t = input_data[:, t]
            a_prev, c_prev, candidate, ft, ut, ot = self.forward_cell(x_t, a_prev, c_prev)
            self.a[:, t] = a_prev
            self.c[:, t] = c_prev
            self.candidate[:, t] = candidate
            self.ft[:, t] = ft
            self.ut[:, t] = ut
            self.ot[:, t]= ot
        
        if self.architecture == RnnArchitecture.many_to_many:
            return self.a
        if self.architecture == RnnArchitecture.many_to_one:
            return a_prev

    def backward(self, dLoss_per_a : np.ndarray):
        # dLoss_per_a shape : (m, na)
        self.time_cache = {'dWf': [], 'dWu': [], 'dWo': [], 'dWc' : [], 'dbf' : [], 'dbu': [], 'dbo': [], 'dbc': []}
        dLoss_per_x = np.zeros_like(self.x)
        dLoss_per_a_prev = dLoss_per_a
        dLoss_per_c_prev = np.zeros_like(dLoss_per_a_prev)
        for t in reversed(range(self.T_x)):
            if self.architecture == RnnArchitecture.many_to_many:
                dLoss_per_x_t, dLoss_per_a_prev = self.backward_cell(dLoss_per_a[:, t], t)
            elif self.architecture == RnnArchitecture.many_to_one:
                dLoss_per_x_t, dLoss_per_a_prev, dLoss_per_c_prev = self.backward_cell(dLoss_per_a_prev, dLoss_per_c_prev, t)
            dLoss_per_x[:, t] = dLoss_per_x_t
        cumul_gradients = [np.sum(self.time_cache[gradient], axis=0) for gradient in self.time_cache.keys()]
        return dLoss_per_x, cumul_gradients
    
    def update_params(self, gradient : np.ndarray):
        self.Wf += gradient[0]
        self.Wu += gradient[1]
        self.Wo += gradient[2]
        self.Wc += gradient[3]
        self.bf += gradient[4]
        self.bu += gradient[5]
        self.bo += gradient[6]
        self.bc += gradient[7]
        return
    
    
class Sequential:
    def __init__(self, usage:Usage = Usage.regression) -> None:
        self.layers : list[Layer] = []
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
        self.layers.append(layer)
    
    def pop_cache(self) -> dict[str, any]:
        return self.cache.pop()
    
    def init_cache(self) -> None:        
        self.cache.clear()
        zeros = [(0, 0) for l in self.layers]        
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
    
    def disable_dropout(self, p: float):
        if not p: return
        for layer in self.layers:
            if layer != self.layers[-1]: layer.scale_kernel(p)

    def compile(self, learning_rate : float | LearningRateScheduler = 0.01, loss_fn: str | Loss = "l2", 
                regularizer: Regularizer=None, dropout: float = None, optimizer: Optimizer=None):        
        self.loss_fn = Loss(loss_fn) if type(loss_fn) != Loss else loss_fn
        self.optimizer = (optimizer if optimizer else Optimizer(learning_rate))
        self.init_cache()
        self.regularizer = regularizer        
        self.dropout = dropout
        if dropout and (n_layers := len(self.layers)) > 1:            
            [self.layers.insert(i, Dropout(dropout)) for i in range(1, n_layers+1, 2)]
        return
    
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
            
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for layer in self.layers:
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
        
        for l_index, layer in enumerate(reversed(self.layers)):
            dLoss_per_output, dLoss_per_params = layer.backward(dLoss_per_output)
            self.cache[-1]['gradients'].insert(0, dLoss_per_params)
        
        for l_index, layer in enumerate(self.layers):
            gradients = self.cache[-1]['gradients'][l_index]
            if gradients is not None:
                # Optimization and regularization            
                # if self.regularizer:
                #    self.cache[-1]['gradients'][l_index][0] += self.regularizer.gradients_reg(layer.kernel) # Attention
                if self.optimizer:
                    gradients = self.optimizer(gradients, l_index, step=self.training_steps_tracker, epoch=len(self.cache)) 
                layer.update_params(gradients)
                self.cache[-1]['gradients'][l_index] = gradients
                    
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
            if self.usage == Usage.multiClassification or self.usage == Usage.logisticRegression:                 
                classes = self.layers[-1].get_classes() if self.usage == Usage.multiClassification else [0, 1]
                eval_metrics['conf_mat'] = (ConfusionMatrix(Y_hat, Y, classes, usage=self.usage, verbose=True)).output
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
        for layer in self.layers:
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
        for layer in self.layers:
            print(layer.name + " (" + type(layer).__name__ + ")" + "\t\t\t", end='')
            print(str(layer.get_params_count())+"\n")
            total += layer.get_params_count()
        print("========================================================\n")
        print("Total params: " + str(total) + "\n\n")
        
    def save(self, model_name):
        # Save the model in the working directory
        with open(f"{model_name}.ssj","wb") as save_file:
            pickle.dump(self, save_file)
        
    def display_losses(self, *args, **kwargs):
        return
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
    

def load(model_name):
    with open(f"{model_name}.ssj","rb") as save_file:
        load = pickle.load(save_file)
        return load


# if __name__ == '__main__':