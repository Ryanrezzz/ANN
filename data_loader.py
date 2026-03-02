import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data():
    (X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
    return (X_train,y_train),(X_test,y_test)

def explore_data(X_train,y_train):
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Image size     :{X_train.shape[1]} X {X_train.shape[2]}")
    print(f"Number of classes : {len(set(y_train))}")

    unique,counts = np.unique(y_train,return_counts=True)
    for cls, cnt in zip(unique,counts):
        print(f"    {CLASS_NAMES[cls]:15s}: {cnt}")

def prepare_data(X_train,X_test):
    X_train_norm = X_train.astype('float32')/255.0
    X_test_norm=X_test.astype('float32')/255.0
    
    X_train_flat = X_train_norm.reshape(X_train_norm.shape[0],-1)
    X_test_flat = X_test_norm.reshape(X_test_norm.shape[0],-1)

    return X_train_norm,X_test_norm,X_train_flat,X_test_flat

def plot_sample(X_train,y_train):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        idx = np.where(y_train == i)[0][0]
        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(CLASS_NAMES[i])
        ax.axis('off')
    plt.suptitle('Fashion MNIST — One Sample Per Class', fontsize=14)
    plt.tight_layout()
    return fig

(X_train, y_train), (X_test, y_test) = load_data() 
explore_data(X_train,y_train)
prepare_data(X_train,X_test)
fig=plot_sample(X_train,y_train)
plt.show()

