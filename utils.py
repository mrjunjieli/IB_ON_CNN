 #load datasets

import keras
import keras.backend as K
import numpy as np
import scipy.io as sio
from collections import namedtuple
import sys
import os
import pickle


def get_data(flag):
    #flag = 0   MNIST data
    #flag = 1   F-MNIST
    #flag = 2   cifar
    comment = dict()
    comment[0] = 'MNIST'
    comment[1] = 'F-MNIST'
    comment[2] = 'cifar'
    
    print('the dataset is ',comment[flag])
    
    #Returns two namedtuples and one inputshape size,with MNIST training and testing data
    #   trn.X is training data 
    #   trn.Y is training class, with numbers 0~9
    #   tst.X is testing data
    #   tst.Y is testing class,with numbers 0~9

    nb_classes = 10
    
    #dataset input channel
    channel = 1
    
    if flag ==0:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        #input image dimensions
        img_rows,img_cols = 28,28
        channel = 1
    elif flag ==1:
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        #input image dimensions
        img_rows,img_cols = 28,28
        channel = 1
    elif flag==2:
        (X_train,y_train),(X_test,y_test) = keras.datasets.cifar10.load_data()
        X_train = (X_train[:,:,:,0]+X_train[:,:,:,1]+X_train[:,:,:,2])/3
        X_test = (X_test[:,:,:,0]+X_test[:,:,:,1]+X_test[:,:,:,2])/3
        #input image dimensions
        img_rows,img_cols = 32,32
        
        channel=1
    else:
        print('flag input ERROR!')
    
    # print(X_train.shape) (60000,28,28)
    
    if K.image_data_format() == 'channels_first':
        #change data format
        X_train = X_train.reshape(X_train.shape[0],channel,img_rows,img_cols)
        X_test = X_test.reshape(X_test.shape[0],channel,img_rows,img_cols)
        input_shape = (channel,img_rows,img_cols)
    else:
        #change data format
        #reshape from (img_rows,img_cols) to (img_rows,img_cols,1)
        X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,channel)
        X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,channel)
        input_shape = (img_rows,img_cols,channel) 
    
    #0~255 to 0~1
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    
    print('input_shape:',input_shape)
    print('X_train shape:{},X_test shape:{}'.format(X_train.shape,X_test.shape))
    
    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)
    
    print('Y_train shape:{1},Y_test shape:{0}'.format(Y_test.shape,Y_train.shape))
    # print(Y_test)
    print('y_train shape:{},y_test shape:{}'.format(y_train.shape,y_test.shape))
    
    
    #Y means binary class matrices
    #y means integer laber
    Dataset = namedtuple('Dataset',['X','Y','y'])
    trn = Dataset(X_train, Y_train,y_train)
    tst = Dataset(X_test , Y_test,y_test )
    
    return trn,tst,input_shape


def get_dataByme(flag):
    print("Load DATA")    
    trn, tst,input_shape = get_data(flag)       
    if (not os.path.exists("DATASET/data0"))or(not os.path.exists("DATASET/data1")) or (not os.path.exists("DATASET/data2")):  
        try:
            os.makedirs("DATASET")
        except Exception as e:
            print(e)
        with open('choice', 'rb') as f:
                label_choice_trn = pickle.load(f) #dict{keys=[0,1,...9],values=}

        trn_X=trn.X[label_choice_trn[0]]
        trn_Y=trn.Y[label_choice_trn[0]]
        trn_y=trn.y[label_choice_trn[0]]


        for i in range(1,10):
            trn_X=np.concatenate((trn_X,trn.X[label_choice_trn[i]]),0)
            trn_Y=np.concatenate((trn_Y,trn.Y[label_choice_trn[i]]),0)
            trn_y=np.concatenate((trn_y,trn.y[label_choice_trn[i]]),0)
            
        trn_new = {'X':trn_X, 'Y':trn_Y,'y':trn_y}
        tst_new = {'X':tst.X, 'Y':tst.Y,'y':tst.y}
        
        Dataset = namedtuple('Dataset',['X','Y','y'])
        trn = Dataset(trn_X, trn_Y,trn_y)
        
        if flag==0:
            with open("DATASET/data0", 'wb') as f:
                    pickle.dump({'trn':trn_new,'tst':tst_new}, f, pickle.HIGHEST_PROTOCOL) 
        elif flag==1:
            with open("DATASET/data1", 'wb') as f:
                    pickle.dump({'trn':trn_new,'tst':tst_new}, f, pickle.HIGHEST_PROTOCOL) 
        elif flag==2:
            with open("DATASET/data2", 'wb') as f:
                    pickle.dump({'trn':trn_new,'tst':tst_new}, f, pickle.HIGHEST_PROTOCOL) 
    else:
        if flag==0:
            with open('DATASET/data0', 'rb') as f:
                temp = pickle.load(f)
                trn_X = temp['trn']['X']
                trn_Y = temp['trn']['Y']
                trn_y = temp['trn']['y']
                Dataset = namedtuple('Dataset',['X','Y','y'])
                trn = Dataset(trn_X, trn_Y,trn_y)
                
                tst_X = temp['tst']['X']
                tst_Y = temp['tst']['Y']
                tst_y = temp['tst']['y']
                tst = Dataset(tst_X,tst_Y,tst_y)
                
        elif flag==1:
            with open('DATASET/data1', 'rb') as f:
                temp = pickle.load(f)
                trn_X = temp['trn']['X']
                trn_Y = temp['trn']['Y']
                trn_y = temp['trn']['y']
                Dataset = namedtuple('Dataset',['X','Y','y'])
                trn = Dataset(trn_X, trn_Y,trn_y)
                
                tst_X = temp['tst']['X']
                tst_Y = temp['tst']['Y']
                tst_y = temp['tst']['y']
                tst = Dataset(tst_X,tst_Y,tst_y)
        elif flag==2:
            with open('DATASET/data2', 'rb') as f:
                temp = pickle.load(f)
                trn_X = temp['trn']['X']
                trn_Y = temp['trn']['Y']
                trn_y = temp['trn']['y']
                Dataset = namedtuple('Dataset',['X','Y','y'])
                trn = Dataset(trn_X, trn_Y,trn_y)
                
                tst_X = temp['tst']['X']
                tst_Y = temp['tst']['Y']
                tst_y = temp['tst']['y']
                tst = Dataset(tst_X,tst_Y,tst_y)
    return trn,tst,input_shape
    


if __name__ == '__main__':
    choice = 0
    trn,tst,input_shape = get_data(choice)
