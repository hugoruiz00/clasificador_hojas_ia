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

def load_data(resize=None):
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    train_dataset = dataset.ImageFolderDataset(training_path, flag=0).transform_first(trans)
    test_dataset = dataset.ImageFolderDataset(testing_path, flag=0).transform_first(trans)

    aux_train = [x for x in train_dataset]
    train_list_dataset = []
    for i in range(int(len(aux_train)/10)):
        indice = i
        for j in range(10):
            train_list_dataset.append(aux_train[indice])
            indice = indice + 25
    return train_list_dataset, test_dataset

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
            gluon.data.DataLoader(test_dataset, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))

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
    return sum(testLoss) / 50

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    #animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 2],
    #                    legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        #animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    test_loss = getTestLoss(net, loss, test_iter)
    #animator.showGraph()
    print(train_loss, train_acc, test_loss, test_acc)
    #assert train_loss < 0.5, train_loss
    #assert train_acc <= 1 and train_acc > 0.7, train_acc
    #assert test_acc <= 1 and test_acc > 0.7, test_acc
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
            ax.set_title(titles[i])
    plt.show()
    return axes

def get_trees_labels(labels):  #@save
    text_labels = [
        'Almendra', 'Benjamina', 'Chaya', 'Encino', 'Guayaba', 'Mora', 'Nispero', 'Pata de vaca', 'Roble', 'Tulipan']
    return [text_labels[int(i)] for i in labels]

def predict_ch3(net, test_iter, n=8):  #@save
    for X, y in test_iter:
        break
    trues = get_trees_labels(y)
    preds = get_trees_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 100, 100)), 1, n, titles=titles[0:n])

def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = len(X) // k
    train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        train_part = X[idx]
        if j == i:
            valid = train_part
        elif train is None:
            train = train_part
        else:
            train.extend(train_part) 
    return train, valid

def k_fold(k, data_train, num_epochs, batch_size, net, loss, trainer):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        fold_train, fold_valid = get_k_fold_data(k, i, data_train)
        train_iter, test_iter = convert_to_dataloader(fold_train, fold_valid, batch_size)
        train_ls, valid_ls = train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
        train_l_sum += train_ls
        valid_l_sum += valid_ls
        #if i == 0:
        #    d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
        #             xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
        #             legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train loss {float(train_ls):f}, '
              f'valid loss {float(valid_ls):f}')
    return train_l_sum / k, valid_l_sum / k

def getNet(numNeurons, FA):
    net = nn.Sequential()
    net.add(nn.Dense(numNeurons, activation=FA), nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    return net

def initialize_parameters(num_neurons, lr, FA):
    net = getNet(num_neurons, FA)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    return net, loss, trainer

def hyperparameters_selection():
    for i in range(num_combinations):
        num_neurons = list_num_neurons[i]
        lr = list_lr[i]
        num_epochs = list_num_epochs[i]
        FA = list_FA[i]
        net, loss, trainer = initialize_parameters(num_neurons, lr, FA)
        train_l, valid_l = k_fold(k, train_list, num_epochs, batch_size, net, loss, trainer)
        errorsTrain.append(train_l)
        errorsValid.append(valid_l)
        print(f'{k}-fold validation: avg train loss: {float(train_l):f}, '
            f'avg valid loss: {float(valid_l):f}')              

def show_results_selection(errorsTrain, errorsValid):
    infoH1 = "1 capa oculta de "+str(list_num_neurons[0])+" neuronas\n0.01 Tasa de aprendizaje\n"+str(list_num_epochs[0])+" épocas"+"\nFA "+list_FA[0]
    infoH2 = "1 capa oculta de "+str(list_num_neurons[1])+" neuronas\n0.001 Tasa de aprendizaje\n"+str(list_num_epochs[1])+" épocas"+"\nFA "+list_FA[1]
    infoH3 = "1 capa oculta de "+str(list_num_neurons[2])+" neuronas\n0.001 Tasa de aprendizaje\n"+str(list_num_epochs[2])+" épocas"+"\nFA "+list_FA[2]
    infoH4 = "1 capa oculta de "+str(list_num_neurons[3])+" neuronas\n0.001 Tasa de aprendizaje\n"+str(list_num_epochs[3])+" épocas"+"\nFA "+list_FA[3]
    infoH5 = "1 capa oculta de "+str(list_num_neurons[4])+" neuronas\n0.0003 Tasa de aprendizaje\n"+str(list_num_epochs[4])+" épocas"+"\nFA "+list_FA[4]
    hyperparameters = ['Hiperparámetros 1\n'+infoH1, 'Hiperparámetros 2\n'+infoH2, 
    'Hiperparámetros 3\n'+infoH3, 'Hiperparámetros 4\n'+infoH4, 'Hiperparámetros 5\n'+infoH5]
    x = arange(len(hyperparameters))
    width = 0.35
    fig, ax = plt.subplots()   
    rects1 = ax.bar(x - width/2, errorsTrain, width, label='Error de entrenamiento')
    rects2 = ax.bar(x + width/2, errorsValid, width, label='Error de validación')
    ax.set_ylabel('Error')
    ax.set_title('Error de entrenamiento y de validación en Validación Cruzada para selección de hiperparámetros')
    ax.set_xticks(x)
    ax.set_xticklabels(hyperparameters, fontsize = 9)
    ax.legend()
    ax.text(2.7, 0.45, "Todos cuentan con una sola capa oculta\ny con la misma inicialización de pesos.\nLos hiperparámetros a optimizar son\nel número épocas, tasa de aprendizaje,\nla FA y el número de neuronas en la\ncapa oculta.", fontsize=9)
    def autolabel(rects):
        for rect in rects:
            height = str(round(rect.get_height(), 5))
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, float(height)),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    #plt.savefig('model-selection.png')
    plt.show()
 
def save_best_hyperparameters():
    indice = errorsValid.index(min(errorsValid))
    hyperparameters = []
    hyperparameters.append(list_num_neurons[indice])
    hyperparameters.append(list_lr[indice])
    hyperparameters.append(list_num_epochs[indice])
    hyperparameters.append(list_FA[indice])
    f = open("hyperparameters.txt", "w")
    for hyperparameter in hyperparameters:
        f.write(str(hyperparameter))
        f.write("\n")   
    f.close()


# Parameters
k, batch_size = 5, 1

#Load data from local folder
train_list, test_list = load_data(100)


# Model selection
errorsTrain = []
errorsValid = []

list_num_neurons = [64, 64, 256, 32, 128]
list_lr = [0.01, 0.001, 0.001, 0.001, 0.0003]
list_num_epochs = [100, 10, 20, 30, 120]
list_FA = ['relu', 'tanh', 'relu', 'relu', 'tanh']
num_combinations = len(list_num_epochs)

hyperparameters_selection()
save_best_hyperparameters()
show_results_selection(errorsTrain, errorsValid)



#test1 = [0.64173701, 0.21838010, 0.19252801, 0.26348401, 0.11650201]
#test2 = [0.65101101, 0.21123601, 0.18963201, 0.29647401, 0.12737601]
#show_results_selection(test1, test2)
