# @Date:   2020-04-24T10:19:05+02:00
# @Last modified time: 2020-04-25T11:59:59+02:00



import os



import numpy as np
import h5py
import scipy.io as scio

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
session = tf.Session(config=config)

def predata4LCZ(file, keyX, keyY):
    hf = h5py.File(file, 'r')
    x_tra = np.array(hf[keyX])
    y_tra = np.array(hf[keyY])
    hf.close()

    print(x_tra.shape, y_tra.shape)

    return x_tra, y_tra
################################################################################
def evaluate(model):
    file0 ='./results/'
    batch_size = 32#8 16 32
    numC= 17 ;

    'loading test data'
    file='../PROJECT_training2005/dlr_challenge/data/validation.h5'
    x_tst, y_tst= predata4LCZ(file, 'sen2', 'label')
    patch_shape = (32, 32, 10)


    #########################################
    modelbest = file0  + "_" + str(batch_size) +"_weights.best.hdf5"


    'load saved best model'
    model.load_weights(modelbest, by_name=False)

    # 4. test phase
    mc_predictions = []
    for i in range(100):
        y_pre = model.predict(x_tst, batch_size = batch_size)
        #y_pre = y_pre.argmax(axis=-1)+1
        mc_predictions.append(y_pre)

    y_testV = y_tst.argmax(axis=-1)+1

    accs = []
    for y_p in mc_predictions:
        acc = accuracy_score(y_testV, y_p.argmax(axis=-1)+1)
        accs.append(acc)

    print("MC accuracy: {:.1%}".format(sum(accs)/len(accs)))

    mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)+1
    ensemble_acc = accuracy_score(y_testV, mc_ensemble_pred)
    print("MC-ensemble accuracy: {:.1%}".format(ensemble_acc))
    
    idx = 10
    
    p0 = np.array([p[idx] for p in mc_predictions])
    print(p0.shape)
    print("posterior mean: {}".format(p0.mean(axis=0).argmax()+1))
    print("true label: {}".format(y_tst[idx].argmax()+1))
    print()
    # probability + variance
    for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
        print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i+1, prob, var))
       
    x, y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
    
    max_vars = []
    for idx in range(len(mc_predictions)):
        px = np.array([p[idx] for p in mc_predictions])
        max_vars.append(px.std(axis=0)[px.mean(axis=0).argmax()])
        
    return accs, ensemble_acc, p0, x, y, max_vars, mc_predictions