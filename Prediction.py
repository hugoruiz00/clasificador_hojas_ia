from mxnet.gluon import nn
from mxnet import autograd, gluon, init, np, npx
from matplotlib import pyplot as plt
import mxnet as mx
import sys

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageOps

npx.set_np()
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

training_path = 'data\\dataset\\train'
testing_path = 'data\\dataset\\test'

def get_dataloader_workers():  
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

def load_data_folder(resize=None): 
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    test_dataset = dataset.ImageFolderDataset(testing_path, flag=0).transform_first(trans)
    return test_dataset

def get_individual_label(label):  #@save
    text_labels = [
        'Almendra', 'Benjamina', 'Chaya', 'Encino', 'Guayaba', 'Mora', 'Nispero', 'Pata de vaca', 'Roble', 'Tulipan']
    return text_labels[int(label)]

def get_parameters():
    f = open('hyperparameters.txt', "r")
    lines = f.readlines()
    f.close()
    lines = [s.rstrip('\n') for s in lines]
    num_neurons = int(lines[0])
    lr = float(lines[1])
    num_epochs = int(lines[2])
    FA = str(lines[3])
    return num_neurons, FA

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

def predict_ch3(net, test_iter, n=12):  #@save
    for X, y in test_iter:
        break
    trues = get_trees_labels(y)
    preds = get_trees_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 100, 100)), 1, n, titles=titles[0:n])

ventana, miFrame = None, None
textURL, txtDescripcion = None, None
textnombrePlanta, hoja = None, None
imagee, gray_image = None, None
isAdded = False

def mostrarVentana():
    global ventana, miFrame, textURL, txtDescripcion, textnombrePlanta, hoja
    ventana = tk.Tk()
    miFrame = tk.Frame(ventana, width=1000, height=550)
    miFrame.configure(background = 'azure3')
    miFrame.pack()
    ventana.configure(background = 'azure3')
    #url de prueba para mostrar una imagen al inicio
    urlImage= r"empty_image.png"        
    
    textURL = tk.StringVar()
    textURL.set("URL IMAGEN")

    textnombrePlanta = tk.StringVar()
    textnombrePlanta.set("Nombre de la planta")

    
    image = ImageTk.PhotoImage( Image.open( urlImage ).resize( (195,345) ) )
    hoja = tk.Label(miFrame, image = image )
    hoja.place(x=20, y=20)

    txtDescripcion = tk.Text(miFrame, width= 60, height=18 )
    txtDescripcion.insert(1.0, "\tInformación")
    txtDescripcion.configure(font=("Courier",12, "italic"))#, state='disabled')
    txtDescripcion.place(x=280, y=20)
    
    txtNombre =  tk.Label(miFrame, textvariable=textnombrePlanta)
    txtNombre.place(x=50, y=390)

    txtURL = tk.Label(miFrame, textvariable=textURL)
    txtURL.place(x=120, y=440)
    
    imgCar = ImageTk.PhotoImage( Image.open( "btnCargar.png").resize( (100,45) ) )
    btnCargar = tk.Button(miFrame, command=add_image , image = imgCar, borderwidth=0,  bg="azure3" )
    btnCargar.place(x=20, y=430)

    imgIdent = ImageTk.PhotoImage( Image.open( "btnIdentificar.png") ) 
    btn = tk.Button(miFrame, command = identificar, image = imgIdent, borderwidth=0,  bg="azure3" )
    btn.place(x=400, y=380)

    imgPredicts = ImageTk.PhotoImage( Image.open( "btnPredicciones.png") ) 
    btnPredict = tk.Button(miFrame, command = showPredicts, image = imgPredicts, borderwidth=0,  bg="azure3" )
    btnPredict.place(x=650, y=380)

    ventana.mainloop()

def add_image():
    global imagee, gray_image, isAdded
    ventana.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    textURL.set( ventana.filename)    
    gray_image = ImageOps.grayscale(Image.open(ventana.filename).resize((100, 100)))
    imagee = ImageTk.PhotoImage( Image.open( ventana.filename ).resize( (195,345) ) )
    hoja = tk.Label(miFrame, image = imagee )
    hoja.place(x=20, y=20)
    txtDescripcion.delete("1.0","end")
    isAdded = True

def identificar():
    if(isAdded):
        txtDescripcion.delete("1.0","end")
        #txtDescripcion.insert(1.0, textURL.get() )
        result = trained_net(np.array(gray_image).reshape(1,100,100)).argmax(axis=1)
        textnombrePlanta.set(get_individual_label(result))
        f = open('infoTrees\\'+get_individual_label(result)+'.txt', "r",encoding='utf-8',
                 errors='ignore')
        lines = f.readlines()
        f.close()
        lines = [s.rstrip('\n') for s in lines]
        text = ""
        for line in lines:
            text = text + line+"\n"
        txtDescripcion.insert(1.0, text )
    else:
        tk.messagebox.showinfo(title="Error", message="No se ha cargado ninguna imagen.\nPresione el botón de 'Cargar imagen'.")

def getNet(numNeurons, FA):
    net = nn.Sequential()
    net.add(nn.Dense(numNeurons, activation=FA), nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    return net

def get_trained_net():
    file_name = "net.params"
    num_neurons, FA = get_parameters()
    trained_net = getNet(num_neurons, FA)
    trained_net.load_parameters(file_name, ctx=ctx)
    return trained_net

def showPredicts():
    test_iter = load_data_folder(100)
    test_data = gluon.data.DataLoader(test_iter, 20, shuffle=True, num_workers=get_dataloader_workers())
    predict_ch3(trained_net, test_data)

trained_net = get_trained_net()
mostrarVentana()
