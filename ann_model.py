import time
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout ,Input
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from data_loader import CLASS_NAMES

def build_ann():
    model =Sequential([
        Input(shape=(28,28)),
        Flatten(),
        Dense(256,activation='relu'),
        Dropout(0.3),
        Dense(128,activation='relu'),
        Dropout(0.2),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_ann(model,X_train,y_train,X_test,y_test,epochs=20):
    start = time.time()
    history = model.fit(X_train,y_train,epochs=epochs,batch_size=128,validation_data=(X_test,y_test),verbose=1)
    train_time=time.time()-start

    y_pred= model.predict(X_test).argmax(axis=1)
    acc = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred,target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test,y_pred)

    return{
        'model':model,
        'accuracy':acc,
        'train_time':train_time,
        'y_pred':y_pred,
        'report':report,
        'confusion_matrix':cm.tolist(),
        'history':history.history
    }
