import keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

from keras.models import load_model
import pickle
import os
from keras.models import Model

import utils

class LoggingReporter(keras.callbacks.Callback):
    def __init__(self, cfg, trn, tst, do_save_func=None, *kargs, **kwargs):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        
        self.cfg = cfg # Configuration options dictionary
        self.trn = trn  # Train data(X,Y,y)
        self.tst = tst  # Test data(X,Y,y )
        
  
        # if we need log this epoch, this function will return True
        self.do_save_func = do_save_func
        
        self.saved_network = './%s/saved_networks'%(cfg['ARCH_NAME'])
        
    def on_train_begin(self, logs={}):
        
        if not os.path.exists(self.cfg['SAVE_DIR']):
            print("Making directory", self.cfg['SAVE_DIR'])
            os.makedirs(self.cfg['SAVE_DIR'])

            
        #the index of the layers that have attribute 'kernel'
        #including Dense or CNN
        #remove the  'Flatten()'
        self.layerixs = []  
    
        # Functions return activity of each layer
        self.layerfuncs = []
        
        # Functions return weights of each layer
        self.layerweights = []
        
        #flatten list of the layers
        for lndx, l in enumerate(self.model.layers):   
            
            if hasattr(l, 'kernel'):
                self.layerixs.append(lndx)
                self.layerfuncs.append(l.output)
                self.layerweights.append(l.kernel)
        print('len(layers):{}'.format(len(self.layerixs)))
        # input_tensors = [self.model.inputs[0],#input data 
        #                  self.model.sample_weights[0],#how much to weight each sample by 
        #                  self.model.targets[0],#labels
        #                  K.learning_phase()]#train or test mode 

        
        # # Get gradients of all the relevant layers at once
        # grads = self.model.optimizer.get_gradients(self.model.total_loss, self.layerweights)

        
        # self.get_gradients = K.function(inputs=input_tensors,outputs=grads)        
        # Get cross-entropy loss
        # self.get_loss = K.function(inputs=input_tensors, outputs=[self.model.total_loss,])
                
    def on_epoch_begin(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            self._log_gradients = False
        else:
            # We will log this epoch.  For each batch in this epoch, we will save the gradients (in on_batch_begin)
            # We will then compute means and vars of these gradients
            
            self._log_gradients = True
            self._batch_weightnorm = []
                
            #self._batch_gradients = [ [] for _ in self.layerixs[:] ]   #layers size - 1
            
            # Indexes of all the training data samples. These are shuffled and read-in in chunks of BATCHSIZE
            ixs = list(range(len(self.trn.X)))
            self._batch_todo_ixs = ixs #shuffle index of train data

    def on_batch_begin(self, batch, logs={}):
        if not self._log_gradients:
            # We are not keeping track of batch gradients, so do nothing
            return
        
        # Sample a batch
        batchsize = self.cfg['BATCHSIZE']
        #cur_ixs = self._batch_todo_ixs[:batchsize]  #get batchsize of traindata
        # Advance the indexing, so next on_batch_begin samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]
        
        # Get gradients for this batch
        # inputs = [self.trn.X[cur_ixs,:],  # Inputs
        #           [1,]*len(cur_ixs),      # Uniform sample weights
        #           self.trn.Y[cur_ixs,:],  # Outputs
        #           1                       # Training phase
        #          ]
        
        # for lndx, g in enumerate(self.get_gradients(inputs)):
        #     # g is gradients for weights of lndx's layer
        #     oneDgrad = g.reshape(-1, 1)                  # Flatten to one dimensional vector
        #     self._batch_gradients[lndx].append(oneDgrad)     


    def on_epoch_end(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return

        # Get overall performance
        loss = {}
        # for cdata, cdataname, istrain in ((self.trn,'trn',1), (self.tst, 'tst',0)):
            # loss[cdataname] = self.get_loss([cdata.X, [1,]*len(cdata.X), cdata.Y, istrain])[0].flat[0]
        loss['trn'],_=self.model.evaluate(self.trn.X,self.trn.Y,verbose=0)
        loss['tst'],_= self.model.evaluate(self.tst.X,self.tst.Y,verbose=0)
            
        data = {
            'weights_norm' : [],   # L2 norm of weights
            'gradmean'     : [],   # Mean of gradients
            'gradstd'      : [],   # Std of gradients
            'activity_tst' : []    # Activity in each layer for test set
        }
        
        for lndx, layerix in enumerate(self.layerixs):

            clayer = self.model.layers[layerix]
            
            #data['weights_norm'].append(np.linalg.norm(K.get_value(clayer.kernel)) )
            
            #stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            # data['gradmean'    ].append( np.linalg.norm(stackedgrads.mean(axis=1)) )
            # data['gradstd'     ].append( np.linalg.norm(stackedgrads.std(axis=1))  )
            if(self.cfg['train_test']=='train'):
                data['activity_tst'].append(Model(inputs = self.model.input,outputs=self.layerfuncs[lndx]).predict(self.trn.X))
            elif(self.cfg['train_test']=='test'):
                data['activity_tst'].append(Model(inputs = self.model.input,outputs=self.layerfuncs[lndx]).predict(self.tst.X))
            # data['activity_tst'].append( self.layerfuncs[lndx]([self.tst.X,])[0]   )
                
        
        fname = self.cfg['SAVE_DIR'] + "/epoch%08d"% epoch
        print("Saving", fname)
        with open(fname, 'wb') as f:
            pickle.dump({'ACTIVATION':self.cfg['ACTIVATION'], 'epoch':epoch, 'data':data,'loss':loss}, f, pickle.HIGHEST_PROTOCOL)        
         
        
        # #save net work model 
        
        self.model.save('%s/%08d.h5'%(self.saved_network,epoch))
        
        model_name_list = sorted(os.listdir(self.saved_network))
        
        if len(model_name_list)>5:
            model_name_remove = model_name_list[1:-3]  #ignore .ipynb file
            for name in model_name_remove:
                os.remove('%s/%s'%(self.saved_network,name))
        