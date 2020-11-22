import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,LocallyConnected2D
from keras import backend as K
import os
import tensorflow as tf 
import json 
#from keras.utils import multi_gpu_model 
import loggingreporter  
import utils



import tensorflow as tf

cfg = {}
cfg['BATCHSIZE']    =1000
cfg['LEARNINGRATE'] = 0.001
cfg['NUM_EPOCHS']  =10000
cfg['output_shape'] = [6,6]
cfg['kernel_size'] =  [3,3]
cfg['num_classes'] = 10
cfg['ACTIVATION'] = 'tanh'
cfg['train_test'] = 'train'
# 0 or 1 0--Mnist 1--Fashion Mnist  2--cifar10
choice = 2
#0 or 1 o--no pooling 1--pooling
pool = 0  

#load data
trn, tst,input_shape = utils.get_data(choice)

#--------------------------------------------------

#cnn model
model = Sequential()
for index,(x,y) in enumerate(zip(cfg['output_shape'],cfg['kernel_size'])):
    if index==0:
        model.add(Conv2D(x,kernel_size=(y,y),activation= cfg['ACTIVATION'],input_shape= input_shape,padding='same'))
    else:
        model.add(Conv2D(x, (y, y), activation=cfg['ACTIVATION']))
        # if (index+1)%2==0:
        #     model.add(MaxPooling2D(padding='same'))
        #     pool = 1

model.add(Flatten())
# model.add(Dense(500, activation=cfg['ACTIVATION']))
# model.add(Dense(1024, activation=cfg['ACTIVATION']))
# model.add(Dense(500, activation=cfg['ACTIVATION']))
model.add(Dense(cfg['num_classes'], activation='softmax'))


#model = multi_gpu_model(model,gpus=2)

#--------------------------------------------------------------
#file name
ARCH_NAME = 'SHAPE_'+'-'.join(map(str,cfg['output_shape']))

ARCH_NAME =ARCH_NAME+'_KERNEL_'+'-'.join(map(str,cfg['kernel_size']))
ARCH_NAME = ARCH_NAME+'_'+str(cfg['num_classes'])

cfg['ARCH_NAME'] = 'data/'+cfg['train_test']+'/'+cfg['ACTIVATION']+'_'+ARCH_NAME+'_'+'choice='+str(choice)+'_'+'pool='+str(pool)


# Where to save activation and weights data
cfg['SAVE_DIR'] = cfg['ARCH_NAME']+'/rawdata'

#--------------------------------------------------------------





#model compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=cfg['LEARNINGRATE']),
              metrics=['accuracy'])
			  
			  
def do_report(epoch):
    # Only log activity for some epochs.
    if epoch < 10:       # Log for all first 20 epochs
        return ((epoch)%2==0)
    elif epoch < 50:    # Then every 10th
        return ((epoch+1) % 5 == 0)
    elif epoch < 200:    # Then every 10th
        return ((epoch+1) % 50 == 0)
    else:
        return ((epoch+1)%500 ==0 )

reporter = loggingreporter.LoggingReporter(cfg=cfg,trn=trn,tst=tst,do_save_func=do_report)

#------------------------------------------------------------

saved_network = '%s/saved_networks'%(cfg['ARCH_NAME'])

if not os.path.exists(saved_network):
    print("Making directory:",saved_network)
    os.makedirs(saved_network)


#load model

LOAD = True
ini_epoch =-1
if LOAD:
    try:
        model_name = sorted(os.listdir(saved_network))[-1]
        model = load_model('%s/%s'%(saved_network,model_name))
        print('load',model_name)
        ini_epoch = int(model_name.split('.')[0])
    except:
        ini_epoch=-1


model.fit(trn.X, trn.Y,
          batch_size=cfg['BATCHSIZE'],
          epochs=cfg['NUM_EPOCHS'],
          verbose=2,
          initial_epoch=ini_epoch+1,
          callbacks  = [reporter,])


# %load_ext autoreload5
# %autoreload 2
import os, pickle
from collections import defaultdict, OrderedDict
import multiprocessing as mp

import numpy as np
import keras.backend as K

import simplebinmi

import utils
trn, tst,inputshape = utils.get_data(choice)



DO_SAVE        = True    # Whether to save plots or just show them

MAX_EPOCHS = cfg['NUM_EPOCHS']      # Max number of epoch for which to compute mutual information measure



# Functions to return upper and lower bounds on entropy of layer activity
# noise_variance = 1e-1                    # Added Gaussian noise variance
# Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
# entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
# entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

# nats to bits conversion factor
nats2bits = 1.0/np.log(2) 

# Save indexes of tests data for each of the output classes
saved_labelixs = {}

if (cfg['train_test'] == 'test'):
    for i in range(10):
        saved_labelixs[i] = np.squeeze(tst.y == i)
    labelprobs = np.mean(tst.Y, axis=0)
elif (cfg['train_test'] == 'train'):
    for i in range(10):
        saved_labelixs[i] = np.squeeze(trn.y == i)
    labelprobs = np.mean(trn.Y, axis=0)


# Directories from which to load saved layer activity
ARCH = cfg['ARCH_NAME']
DIR_TEMPLATE = '%s/%%s'%ARCH

activation = 'tanh'


cur_dir = DIR_TEMPLATE%('rawdata')


# Data structure used to store results
measures = OrderedDict()
measures[activation] = {}

PLOT_LAYERS =[] 
#which layer to plot
with open(cur_dir+'/epoch00000000', 'rb') as f:
    d = pickle.load(f)
    num_layers = len(d['data']['activity_tst'])
    PLOT_LAYERS = [x for x in range(num_layers)]
print('layers to plot :',PLOT_LAYERS)    

drive_dir = cfg['ARCH_NAME']+'/results'

if not os.path.exists(drive_dir):
    print("Making directory:",drive_dir)
    os.makedirs(drive_dir)
    
print('process file path:',cur_dir)
print('path to save image:',drive_dir)


# #save net config
# import json
# str = json.loads(model.to_json())['config']['layers']
# f = open(drive_dir+'/README.md','w')
# f.write('net model\n')
# for i in range(len(PLOT_LAYERS)+1):
#     f.write(json.dumps(str[i]))
#     f.write('\n')
#     f.write('\n')
# f.close()


#compute MI
#

# Load files saved during each epoch, and compute MI measures of the activity in that epoch
print('*** Doing %s ***' % cur_dir)


for epochfile in sorted(os.listdir(cur_dir)):
    
    #ignore not pickle file
    if not epochfile.startswith('epoch'):
        continue

    fname = cur_dir + "/" + epochfile
    with open(fname, 'rb') as f:
        d = pickle.load(f)

    epoch = d['epoch']

    print("Doing", fname)

    num_layers = len(d['data']['activity_tst'])


    cepochdata = defaultdict(list)
    for lndx in range(num_layers):
        activity = d['data']['activity_tst'][lndx]
        activity = activity.reshape(len(activity),len(activity[0].flatten()))

        # # Compute marginal entropies
        # HM_upper = entropy_func_upper([activity,])[0]         #entropy
        # HM_lower = entropy_func_lower([activity,])[0]         #entropy
        # # Layer activity given input. This is simply the entropy of the Gaussian noise
        # MI_MX = kde.kde_condentropy(activity, noise_variance) #conditional entropy H(M|X)

        
        # # Compute conditional entropies of layer activity given output
        # #H(M|Y)
        # MI_MY_upper=0.
        # for i in range(10):
        #     HMcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
        #     MI_MY_upper += labelprobs[i] * HMcond_upper
        # MI_MY_lower=0.
        # for i in range(10):
        #     HMcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
        #     MI_MY_lower += labelprobs[i] * HMcond_lower
        
        
        # #upper bound
        # cepochdata['MI_XM_upper'].append( nats2bits * (HM_upper - MI_MX) )
        # cepochdata['MI_YM_upper'].append( nats2bits * (HM_upper - MI_MY_upper) )
        # cepochdata['H_M_upper'  ].append( nats2bits * HM_upper )
        # pstr = '\nupper: HM=%0.3f, H(M|X)=%0.3f, MI(X;M)=%0.3f ,H(M|Y)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['H_M_upper'][-1],MI_MX*nats2bits,cepochdata['MI_XM_upper'][-1], MI_MY_upper*nats2bits,cepochdata['MI_YM_upper'][-1])
        
        
        # #lower bound
        # cepochdata['MI_XM_lower'].append( nats2bits * (HM_lower - MI_MX) )
        # cepochdata['MI_YM_lower'].append( nats2bits * (HM_lower - MI_MY_lower) )
        # cepochdata['H_M_lower'  ].append( nats2bits * HM_lower )
        # pstr += '\nlower: HM=%0.3f, H(M|X)=%0.3f ,MI(X;M)=%0.3f ,H(M|Y)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['H_M_lower'][-1],MI_MX*nats2bits,cepochdata['MI_XM_lower'][-1], MI_MY_upper*nats2bits,cepochdata['MI_YM_lower'][-1])
        
        
        
        #bin
        #inputdata = tst.X.reshape(tst.X.shape[0],tst.X.shape[1]*tst.X.shape[2]*tst.X.shape[3])
        if (cfg['train_test'] == 'test'):
            inputdata = tst.X.reshape(tst.X.shape[0],tst.X.shape[1]*tst.X.shape[2]*tst.X.shape[3])
        elif (cfg['train_test'] == 'train'):
            inputdata = trn.X.reshape(trn.X.shape[0],trn.X.shape[1]*trn.X.shape[2]*trn.X.shape[3])

        # inputdata = trn.X.reshape(trn.X.shape[0],trn.X.shape[1]*trn.X.shape[2]*trn.X.shape[3])
        #HM,I(m,x),H(M|X),I(Y,M),H(M|Y)
        binHM,binXM,binX_M, binYM,binY_M = simplebinmi.bin_calc_information2(inputdata,saved_labelixs, activity, 0.67)   
        cepochdata['MI_XM_bin'].append( nats2bits * binXM )
        cepochdata['MI_YM_bin'].append( nats2bits * binYM )
        cepochdata['H_M_bin'].append(nats2bits * binHM)
        pstr = '\nbin  : HM=%0.3f, H(M|X) = %0.3f, MI(X;M)=%0.3f, H(M|Y)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['H_M_bin'][-1],binX_M*nats2bits,cepochdata['MI_XM_bin'][-1],binY_M*nats2bits, cepochdata['MI_YM_bin'][-1])

        
        print('- Epoch %d Layer %d %s' % (epoch,lndx, pstr) )

    measures[activation][epoch] = cepochdata

#保存变量measures    

measures_json = json.dumps(measures)
with open(drive_dir+'/measures.txt', 'w') as f:     
        f.write(measures_json)  



#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')



plt.figure(figsize=(4,8))
gs = gridspec.GridSpec(2,len(measures))
for activation, vals in measures.items():
    epochs = sorted(vals.keys())
          

    #plot bin I(X,M)
    plt.subplot(gs[0,0])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
        plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
#     plt.ylim([0,15])
    plt.xlabel('Epoch')
    plt.ylabel('I(X;T)')

    
    #plot bin I(X,M)
    plt.subplot(gs[1,0])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        hbinnedvals = np.array([vals[epoch]['MI_YM_bin'][layerid] for epoch in epochs])
        plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
#     plt.ylim([0,5])
    plt.xlabel('Epoch')
    plt.ylabel('I(Y;T)')

    
    plt.legend(loc='lower right')
        
plt.tight_layout()


if DO_SAVE:
    plt.savefig(drive_dir+'/MI',bbox_inches='tight')


