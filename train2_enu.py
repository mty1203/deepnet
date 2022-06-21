from cProfile import label
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import time
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from deepsphere import HealpyGCNN
# from deepsphere import healpy_layers as hp_layer
from healpy_networks import HealpyGCNN
import healpy_layers as hp_layer
import pandas as pd
from sklearn.metrics import recall_score
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2","/gpu:3"])

#"/gpu:2","/gpu:3"
font = {'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.98
    self.model.optimizer.lr.assign(new_lr)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoint_theta.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Draw the performance of the model
# x axis: true value
# y axis: predicted value
def draw_performance(x,y,title):
  plt.clf()
  plt.figure(figsize=(12,8))
  plt.grid()
  plt.scatter(x,y, label="predictions", color='red')
  mi = np.amin(x)
  ma = np.amax(x)
  plt.plot([np.amin(x), np.amax(x)], [np.amin(x), np.amax(x)], 'k-', alpha=0.75, zorder=0, label="y=x")
  plt.legend()
  plt.xlabel("True")
  plt.ylabel("Predicted")
  plt.title(title)
  plt.savefig("%s.png" %title)

def get_roc(y, y_hat_prob):
  thresholds = sorted(set(y_hat_prob), reverse=True)
  ret = [[0, 0]]
  for threshold in thresholds:
    y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
    ret.append([get_tpr(y, y_hat), 1 - get_tnr(y, y_hat)])
  return ret

def get_tpr(y, y_hat):
  true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
  actual_positive = sum(y)
  return true_positive / actual_positive

def get_auc(y, y_hat_prob):
  roc = iter(get_roc(y, y_hat_prob))
  tpr_pre, fpr_pre = next(roc)
  auc = 0
  for tpr, fpr in roc:
    auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
    tpr_pre = tpr
    fpr_pre = fpr
  return auc

def get_tnr(y, y_hat):
  true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
  actual_negative = len(y) - sum(y)
  return true_negative / actual_negative
  
def draw_confusion(x,y):
  plt.clf()
  plt.figure(figsize=(12,8))
  plt.grid()
  true_label=x
  predict=y
  confusion = confusion_matrix(true_label,predict)
  plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Oranges)
  plt.title('confusion_matrix')
  plt.colorbar()
  classes=["muon","electron"]
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=-45)
  plt.yticks(tick_marks, classes)
  #ij配对，遍历矩阵迭代器
  iters = np.reshape([[[i,j] for j in range(2)] for i in range(2)],(confusion_matrix.size,2))
  for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]))   #显示对应的数字 
  plt.ylabel("Real Label")
  plt.xlabel("Prediction")
  plt.tight_layout()
  plt.savefig("confusion_matrix.png")
def npe(data):
  npe = np.sum(data,axis=1)
  npe_index = []
  for i in range(npe.shape[0]):
    if npe[i]<5000:
      npe_index.append(False)
    else:
      npe_index.append(True)
  return npe_index
if __name__ == "__main__":
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  ### Init healpix related vars
  nside = 32
  npix = hp.nside2npix(nside)
  indices = np.arange(hp.nside2npix(nside))
  label = []
  x_fht_0 = np.load("/home/duyang/meng/dataset/numu/x_fht_1.npy")
  x_fht_4 = np.load("/home/duyang/meng/dataset/numu/x_fht_2.npy")
  x_fht_1 = np.load("/home/duyang/meng/dataset/nue/x_fht_1.npy")
  x_fht_3 = np.load("/home/duyang/meng/dataset/nue/x_fht_2.npy")
  x_fht_5 = np.load("/home/duyang/meng/dataset/nue/x_fht_3.npy")
  x_fht_6 = np.load("/home/duyang/meng/dataset/nue/x_fht_4.npy")
  x_fht_2 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_fht_1.npy")
  x_fht_7 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_fht_2.npy")
  for i in range(x_fht_0.shape[0]):
    label.append('numu')
  for i in range(x_fht_1.shape[0]):
    label.append('nue')
  for i in range(x_fht_2.shape[0]):
    label.append('nc')
  for i in range(x_fht_3.shape[0]):
    label.append('nue')
  for i in range(x_fht_4.shape[0]):
    label.append('numu')
  for i in range(x_fht_5.shape[0]):
    label.append('nue')
  for i in range(x_fht_6.shape[0]):
    label.append('nue')
  for i in range(x_fht_7.shape[0]):
    label.append('nc')
  x_fht = np.concatenate((x_fht_0,x_fht_1,x_fht_2,x_fht_3,x_fht_4,x_fht_5,x_fht_6,x_fht_7), axis=0)
  del x_fht_0
  del x_fht_1
  del x_fht_2
  del x_fht_3
  del x_fht_4
  del x_fht_5
  del x_fht_6
  del x_fht_7
  x_npe_0 = np.load("/home/duyang/meng/dataset/numu/x_npe_1.npy")
  x_npe_4 = np.load("/home/duyang/meng/dataset/numu/x_npe_2.npy")
  x_npe_1 = np.load("/home/duyang/meng/dataset/nue/x_npe_1.npy")
  x_npe_3 = np.load("/home/duyang/meng/dataset/nue/x_npe_2.npy")
  x_npe_5 = np.load("/home/duyang/meng/dataset/nue/x_npe_3.npy")
  x_npe_6 = np.load("/home/duyang/meng/dataset/nue/x_npe_4.npy")
  x_npe_2 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_npe_1.npy")
  x_npe_7 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_npe_2.npy")
  x_npe = np.concatenate((x_npe_0,x_npe_1,x_npe_2,x_npe_3,x_npe_4,x_npe_5,x_npe_6,x_npe_7), axis=0)
  cut = npe(x_npe)
  del x_npe_0
  del x_npe_1
  del x_npe_2
  del x_npe_3
  del x_npe_4
  del x_npe_5
  del x_npe_6
  del x_npe_7
  x_slope_0 = np.load("/home/duyang/meng/dataset/numu/x_slope_1.npy")
  x_slope_4 = np.load("/home/duyang/meng/dataset/numu/x_slope_2.npy")
  x_slope_1 = np.load("/home/duyang/meng/dataset/nue/x_slope_1.npy")
  x_slope_3 = np.load("/home/duyang/meng/dataset/nue/x_slope_2.npy")
  x_slope_5 = np.load("/home/duyang/meng/dataset/nue/x_slope_3.npy")
  x_slope_6 = np.load("/home/duyang/meng/dataset/nue/x_slope_4.npy")
  x_slope_2 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_slope_1.npy")
  x_slope_7 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_slope_2.npy")
  x_slope = np.concatenate((x_slope_0,x_slope_1,x_slope_2,x_slope_3,x_slope_4,x_slope_5,x_slope_6,x_slope_7), axis=0)
  del x_slope_0
  del x_slope_1
  del x_slope_2
  del x_slope_3
  del x_slope_4
  del x_slope_5
  del x_slope_6
  del x_slope_7
  x_nperatio_0 = np.load("/home/duyang/meng/dataset/numu/x_nperatio_1.npy")
  x_nperatio_4 = np.load("/home/duyang/meng/dataset/numu/x_nperatio_2.npy")
  x_nperatio_1 = np.load("/home/duyang/meng/dataset/nue/x_nperatio_1.npy")
  x_nperatio_3 = np.load("/home/duyang/meng/dataset/nue/x_nperatio_2.npy")
  x_nperatio_5 = np.load("/home/duyang/meng/dataset/nue/x_nperatio_3.npy")
  x_nperatio_6 = np.load("/home/duyang/meng/dataset/nue/x_nperatio_4.npy")
  x_nperatio_2 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_nperatio_1.npy")
  x_nperatio_7 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_nperatio_2.npy")
  x_nperatio = np.concatenate((x_nperatio_0,x_nperatio_1,x_nperatio_2,x_nperatio_3,x_nperatio_4,x_nperatio_5,x_nperatio_6,x_nperatio_7), axis=0)
  del x_nperatio_0
  del x_nperatio_1
  del x_nperatio_2
  del x_nperatio_3
  del x_nperatio_4
  del x_nperatio_5
  del x_nperatio_6
  del x_nperatio_7
  x_mediantime_0 = np.load("/home/duyang/meng/dataset/numu/x_mediantime_1.npy")
  x_mediantime_4 = np.load("/home/duyang/meng/dataset/numu/x_mediantime_2.npy")
  x_mediantime_1 = np.load("/home/duyang/meng/dataset/nue/x_mediantime_1.npy")
  x_mediantime_3 = np.load("/home/duyang/meng/dataset/nue/x_mediantime_2.npy")
  x_mediantime_5 = np.load("/home/duyang/meng/dataset/nue/x_mediantime_3.npy")
  x_mediantime_6 = np.load("/home/duyang/meng/dataset/nue/x_mediantime_4.npy")
  x_mediantime_2 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_mediantime_1.npy")
  x_mediantime_7 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_mediantime_2.npy")
  x_mediantime = np.concatenate((x_mediantime_0,x_mediantime_1,x_mediantime_2,x_mediantime_3,x_mediantime_4,x_mediantime_5,x_mediantime_6,x_mediantime_7), axis=0)
  del x_mediantime_0
  del x_mediantime_1
  del x_mediantime_2
  del x_mediantime_3
  del x_mediantime_4
  del x_mediantime_5
  del x_mediantime_6
  del x_mediantime_7
  # x_slope4_0 = np.load("/home/duyang/meng/dataset/numu/x_slope4_1.npy")
  # x_slope4_4 = np.load("/home/duyang/meng/dataset/numu/x_slope4_2.npy")
  # x_slope4_1 = np.load("/home/duyang/meng/dataset/nue/x_slope4_1.npy")
  # x_slope4_3 = np.load("/home/duyang/meng/dataset/nue/x_slope4_2.npy")
  # x_slope4_5 = np.load("/home/duyang/meng/dataset/nue/x_slope4_3.npy")
  # x_slope4_6 = np.load("/home/duyang/meng/dataset/nue/x_slope4_4.npy")
  # x_slope4_2 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_slope4_1.npy")
  # x_slope4_7 = np.load("/disk_pool/juno_data/dataset_npy/elec_32/nc/x_slope4_2.npy")
  # x_slope4 = np.concatenate((x_slope4_0,x_slope4_1,x_slope4_2,x_slope4_3,x_slope4_4,x_slope4_5,x_slope4_6,x_slope4_7), axis=0)
  # del x_slope4_0
  # del x_slope4_1
  # del x_slope4_2
  # del x_slope4_3
  # del x_slope4_4
  # del x_slope4_5
  # del x_slope4_6
  # del x_slope4_7
  x_all = np.stack((x_fht,x_npe,x_slope,x_nperatio,x_mediantime),axis=-1)
  print(x_all.shape)
  x_all =x_all[cut]
  label = np.array(label)
  label =label[cut]
  label =list(label)
  # x_fht = pd.read_csv("/disk_pool/juno_data/dataset_txt/elecsim/neutrino/4/ElecNu_FHT.csv",dtype=np.float,delimiter=',',header=None,nrows=70000).to_numpy()
  # x_fht[x_fht==1250]=0
  # x_all =x_fht[:,:,np.newaxis]
  # del x_fht
  # x_npe = pd.read_csv("/disk_pool/juno_data/dataset_txt/elecsim/neutrino/4/ElecNu_NPE.csv",dtype=np.float,delimiter=',',header=None,nrows=70000).to_numpy()
  # index = npe(x_npe)
  # x_all = np.concatenate((x_all,x_npe[:,:,np.newaxis]),axis=2)
  # del x_npe
  # x_slope = pd.read_csv("/disk_pool/juno_data/dataset_txt/elecsim/neutrino/4/ElecNu_SLOPE.csv",dtype=np.float,delimiter=',',header=None,nrows=70000).to_numpy()
  # x_all = np.concatenate((x_all,x_slope[:,:,np.newaxis]),axis=2)
  # del x_slope
  # x_peak = pd.read_csv("/disk_pool/juno_data/dataset_txt/elecsim/neutrino/4/ElecNu_PEAK.csv",dtype=np.float,delimiter=',',header=None,nrows=70000).to_numpy()
  # x_all = np.concatenate((x_all,x_peak[:,:,np.newaxis]),axis=2)
  # del x_peak
  # x_nperatio = pd.read_csv("/disk_pool/juno_data/dataset_txt/elecsim/neutrino/4/ElecNu_NPERATIO.csv",dtype=np.float,delimiter=',',header=None,nrows=70000).to_numpy()
  # x_all = np.concatenate((x_all,x_nperatio[:,:,np.newaxis]),axis=2)
  # del x_nperatio
  # x_all =np.delete(x_all,index)
  # y_all = pd.read_csv("/disk_pool/juno_data/dataset_txt/elecsim/neutrino/4/ElecNu_Y.csv",dtype=np.float,delimiter=',',header=None,nrows=70000).to_numpy()
  # y_all = np.delete(y_all,index)
  

  # x_train = np.load("/disk_pool/juno_data/dataset_npy/elec_32/numu/merge_4features/x_train.npy")
  # x_test = np.load("/disk_pool/juno_data/dataset_npy/elec_32/numu/merge_4features/x_test.npy")
  # y_train_all = np.load("/disk_pool/juno_data/dataset_npy/elec_32/numu/merge_4features/y_train.npy")
  # y_test_all = np.load("/disk_pool/juno_data/dataset_npy/elec_32/numu/merge_4features/y_test.npy")
  #y_test =[]
  # for i in range(y_test_all[:,7].shape[0]):
  #   if y_test_all[i][7]!=1:
  #     y_test.append('nc')
  #   else:
  #     if y_test_all[i][6]==12 or y_test_all[i][6]==-12:
  #       y_test.append('nue')
  #     else:
  #       y_test.append('numu')
  print('train_dataset has nc:{} numu:{} nue:{}'.format(label.count('nc'),label.count('numu'),label.count('nue')))
  #,xe_peaktime,xe_MEDIANTIME,xe_MEANTIME,xe_LEPD,xe_NUD,xe_NUV,xe_PEAK,xe_RMS
  ### Split the training set and validation set (8:2)
  enc = OneHotEncoder(sparse=False)
  labels = np.array([label]).T
  label = enc.fit_transform(labels)

 # dataset = tf.data.Dataset.from_tensor_slices((x_all, y_label)).shuffle(70000).batch(32,drop_remainder=True)
 # dataset.cache('./')
  #del x_all
  #train_size =int(0.8*data_size)
  #test_size = int(0.2*data_size)
  #train_dataset = dataset.take(train_size)
  #test_dataset = dataset.skip(train_size)Monomial
  #test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  x_train, x_test, y_train, y_test = train_test_split(x_all, label, test_size=0.20, random_state=42,shuffle=True)
  del x_all
  print("###########################data loaded##############################")
  # layers = [tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=True),
  #           hp_layer.HealpyBernstein(K=8, Fout=12, use_bias=False, use_bn=True, activation=None),
  #           hp_layer.HealpyBernstein(K=8, Fout=24, use_bias=True, use_bn=True, activation="relu"),#10k
  #            #hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 6, "Fout":24,"activation": None,"use_bn": True, "use_bias": True}, activation='relu',use_bn=True),
  #           hp_layer.HealpyPool(p=1),
  #           hp_layer.HealpyBernstein(K=8, Fout=36, use_bias=True, use_bn=True, activation="relu"), #3072
  #           #hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 5, "Fout":48,"activation": None,"use_bn": True, "use_bias": True}, activation='relu',use_bn=True),
  #           hp_layer.HealpyBernstein(K=8, Fout=48, use_bias=True, use_bn=True, activation="relu"),
  #           hp_layer.HealpyPool(p=1),
  #           hp_layer.HealpyBernstein(K=8, Fout=36, use_bias=True, use_bn=True, activation="relu"), #768
  #           hp_layer.HealpyBernstein(K=8, Fout=64, use_bias=True, use_bn=True, activation="relu"),#hp_layer.Healpy_ResidualLayer(layer_type="MONO", layer_kwargs={"K": 5, "Fout":64,"activation": None,"use_bn": True, "use_bias": True}, activation='relu',use_bn=True),
  #           #hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 5, "Fout":64,"activation": None,"use_bn": True, "use_bias": True}, activation='relu',use_bn=True),
  #           hp_layer.HealpyPool(p=1),
  #           #hp_layer.HealpyBernstein(K=10, Fout=48, use_bias=True, use_bn=True, activation="relu"),#768
  #           hp_layer.HealpyBernstein(K=8, Fout=96, use_bias=True, use_bn=True, activation="relu"),#192
  #           hp_layer.HealpyBernstein(K=8, Fout=128, use_bias=True, use_bn=True, activation="relu"),
  #           hp_layer.HealpyPool(p=1),
  #           #hp_layer.HealpyBernstein(K=5, Fout=96, use_bias=True, use_bn=True,activation=None), #48
  #           hp_layer.HealpyBernstein(K=8, Fout=160, use_bias=True, use_bn=True,activation="relu"),  
  #           hp_layer.HealpyPool(p=1),
  #           tf.keras.layers.GlobalAveragePooling1D(),
  #           #tf.keras.layers.Flatten(),    
  #           tf.keras.layers.Dense(3,activation="softmax")]
  # layers = [hp_layer.HealpyGCNNII(layer_kwargs = {},alpha = 0.2),
  #          # tf.keras.layers.Flatten(),
  #          tf.keras.layers.GlobalAveragePooling1D(),
  #           tf.keras.layers.Dense(3,activation="softmax")]
  layers = [tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=True),
            hp_layer.HealpyChebyshev(K=10, M = 12288,Fout=24, use_bias=False, use_bn=True, activation='relu',use_se=True),
            #hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),#10k
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 10, "M":12288,"Fout":24,"activation": None,"use_bn": True, "use_bias": True, "use_se": True}, activation='relu',use_bn=True),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, M = 3072,Fout=48, use_bias=True, use_bn=True, activation="relu",use_se=True), #3072
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 10, "M":3072,"Fout":48,"activation": None,"use_bn": True, "use_bias": True, "use_se": True}, activation='relu',use_bn=True),
            #hp_layer.HealpyChebyshev(K=8, Fout=48, use_bias=True, use_bn=True, activation="relu"),
            hp_layer.HealpyPool(p=1),
            hp_layer.HealpyChebyshev(K=10, M = 768,Fout=64, use_bias=True, use_bn=True, activation="relu",use_se=True), #768
            hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 10, "M":768,"Fout":64,"activation": None,"use_bn": True, "use_bias": True, "use_se": True}, activation='relu',use_bn=True),
            #hp_layer.Healpy_ResidualLayer(layer_type="CHEBY", layer_kwargs={"K": 5, "Fout":64,"activation": None,"use_bn": True, "use_bias": True}, activation='relu',use_bn=True),
            hp_layer.HealpyPool(p=1),
            #hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True, activation="relu"),#768
            hp_layer.HealpyChebyshev(K=10,  M = 192,Fout=96, use_bias=True, use_bn=True, activation="relu",use_se=True),#192
            hp_layer.HealpyChebyshev(K=10, M = 192,Fout=128, use_bias=True, use_bn=True, activation="relu",use_se=True),
            hp_layer.HealpyPool(p=1),
            #hp_layer.HealpyChebyshev(K=5, Fout=96, use_bias=True, use_bn=True,activation=None), #48
            hp_layer.HealpyChebyshev(K=10, M = 48, Fout=160, use_bias=True, use_bn=True,activation="relu",use_se=True),  
            hp_layer.HealpyPool(p=1),
            tf.keras.layers.GlobalAveragePooling1D(),
            #tf.keras.layers.Flatten(),    
            tf.keras.layers.Dense(3,activation="softmax")]
  tf.keras.backend.clear_session()
  with strategy.scope():
    model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=40)
    batch_size = 32
    model.build(input_shape=(None, len(indices), 5))
    model.summary(110)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0006),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  ### Load pre-trained model (if existed).
  if os.path.exists('checkpoint'):
    model.load_weights('weights_classifier.h5')

  else:
    ### Fit the model (if no pre-trained model is found).
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[LearningRateReducerCb(),model_checkpoint_callback]
        )
    print(history.history.keys())
    ### Save the trained model


    ### Draw the learning curve
    plt.figure(figsize=(12,8))
    plt.plot(history.history["loss"], label="training")
    plt.plot(history.history["val_loss"], label="validation")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(" Error")
    plt.savefig("learning_curve_theta.png")
    
    plt.clf()
    plt.figure(figsize=(12,8))
    acc = history.history['accuracy']     #获取训练集准确性数据
    val_acc = history.history['val_accuracy']
    epochs = range(1,len(acc)+1)
    plt.plot(epochs,acc,label='Trainning acc')
    plt.plot(epochs,val_acc,label='Vaildation acc') #以epochs为横坐标，以验证集准确性为纵坐标
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy on Test Set")
    plt.savefig("acc.png")

  model.load_weights('checkpoint_theta.h5')
  predictions = model.predict(x_test)
  init_lables = enc.inverse_transform(predictions)
  np.save("predict.npy", init_lables)
  y_init = enc.inverse_transform(y_test)
  np.save("true.npy", y_init)
  print(y_init)
  c = []
  for item in y_init:
    c.append(item[0])
  #prelabel = []
  classes = list(set(c))
  plt.clf()
  classes.sort()
  confusion = confusion_matrix(init_lables,y_init)
  plt.imshow(confusion, cmap=plt.cm.Blues)
  indices1 = range(len(confusion))
  plt.xticks(indices1, classes)
  plt.yticks(indices1, classes)
  plt.colorbar()
  plt.xlabel('predict')
  plt.ylabel('fact')
  for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
      plt.text(first_index, second_index, confusion[second_index][first_index])
  plt.savefig("matrix.png")
  #print(predictions.shape)
  #points = get_roc(y_test, [x for y in predictions for x in y])
  #points=np.array(points)
  #dtpr = points[:,0]
  #dfpr = points[:,1]
  #df = pd.DataFrame(points, columns=["tpr", "fpr"])
  #print("AUC is %.3f." % get_auc(y_test, prelabel))
 # plt.clf()
  #plt.plot(dfpr, dtpr, label="roc")
 # plt.xlabel("fpr")
  #plt.ylabel("tpr")
  #plt.savefig("roc-1.png")
  ### Save the evaluation results for further analysis
  #seed = 42
  #np.random.seed(seed)
  #kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
  #results = cross_val_score(model,x_input,y_label,scoring="accuracy",cv=kfold)

  
  #draw_performance(y_train, predictions, 'prediction_theta_train')
