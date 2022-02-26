from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
import sys
from Acumulator import Accumulator
from Animator import Animator
from matplotlib import pyplot as plt
import random
from numpy import arange

npx.set_np()

training_path = 'data\\dataset\\train'
testing_path = 'data\\dataset\\test'

def convert_to_dataloader(train, valid, batch_size):
    return (gluon.data.DataLoader(train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(valid, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))

def load_data_folder(batch_size, resize=None): 
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    train_dataset = dataset.ImageFolderDataset(training_path, flag=0).transform_first(trans)
    test_dataset = dataset.ImageFolderDataset(testing_path, flag=0).transform_first(trans)
    return (gluon.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            test_dataset)

def get_dataloader_workers():  
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

def accuracy(y_hat, y):  
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)

    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

def getTestLoss(net, loss, test_iter):
    testLoss = []
    for X, y in test_iter:
        los = loss(net(X), y)
        testLoss.append(sum(los))
    return sum(testLoss) / 30

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    animator = Animator(xlabel='Época', xlim=[1, num_epochs], ylim=[0.1, 2],
                        legend=['Error entrenamiento', 'Precisión entrenamiento', 'Precisión prueba'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print("ACC_",test_acc)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    test_loss = getTestLoss(net, loss, test_iter)
    animator.showGraph()
    print(train_loss, train_acc, test_loss, test_acc)
    return train_loss, test_loss

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontSize=10)
    plt.show()
    return axes

def get_trees_labels(labels):  #@save
    text_labels = [
        'Almendra', 'Benjamina', 'Chaya', 'Encino', 'Guayaba', 'Mora', 'Nispero', 'Pata de vaca', 'Roble', 'Tulipan']
    return [text_labels[int(i)] for i in labels]

def get_individual_label(label):  #@save
    text_labels = [
        'Almendra', 'Benjamina', 'Chaya', 'Encino', 'Guayaba', 'Mora', 'Nispero', 'Pata de vaca', 'Roble', 'Tulipan']
    return text_labels[int(label)]

def predict_ch3(net, test_iter, n=12):  #@save
    for X, y in test_iter:
        break
    trues = get_trees_labels(y)
    preds = get_trees_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 100, 100)), 1, n, titles=titles[0:n])

def getNet(numNeurons, FA):
    net = nn.Sequential()
    net.add(nn.Dense(numNeurons, activation=FA), nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    return net

def get_parameters():
    f = open('hyperparameters.txt', "r")
    lines = f.readlines()
    f.close()
    lines = [s.rstrip('\n') for s in lines]
    num_neurons = int(lines[0])
    lr = float(lines[1])
    num_epochs = int(lines[2])
    FA = str(lines[3])
    return num_neurons, lr, num_epochs, FA

def initialize_parameters(num_neurons, lr, FA):
    net = getNet(num_neurons, FA)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    return net, loss, trainer

# Parameters
batch_size = 1
# Load data from local folder
train_iter, test_iter = load_data_folder(batch_size, 100)
_test_iter = gluon.data.DataLoader(test_iter, batch_size, shuffle=False, num_workers=get_dataloader_workers())

## Creo que es de 150 epocas tanh 84 y 0.00022
## 120 128 y 0.0003
# Train
num_neurons, lr, num_epochs, FA = get_parameters()
print(num_neurons, lr, num_epochs)
net, loss, trainer = initialize_parameters(num_neurons, lr, FA)
train_ch3(net, train_iter, _test_iter, loss, num_epochs, trainer)

# Prediction
test_data = gluon.data.DataLoader(test_iter, 20, shuffle=True, num_workers=get_dataloader_workers())
predict_ch3(net, test_data)

# Save model
file_name = "net.params"
net.save_parameters(file_name)
