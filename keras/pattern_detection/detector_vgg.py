import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,concatenate
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import CSVLogger,TensorBoard
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
import time

##========== data loading ==========#
anode = np.load("../../data/cell_a_MAIKo.npy")
anode = anode.reshape(
    (-1,
     anode.shape[1],
     anode.shape[2],
     1))
cross_point = np.concatenate(
    (np.load("../../data/point_xv_MAIKo.npy"),
     np.load("../../data/point_xs_MAIKo.npy")),
    axis=1)
cross_point = np.concatenate(
    (cross_point[:,0:1],
     cross_point[:,2:3],
     cross_point[:,3:4],
     cross_point[:,5:6]),
    axis=1)
anode_test = np.load("../../data/cell_a_MAIKo_test.npy")
anode_test = anode_test.reshape(
    (-1,
     anode_test.shape[1],
     anode_test.shape[2],
     1))
cross_point_test = np.concatenate(
    (np.load("../../data/point_xv_MAIKo_test.npy"),
     np.load("../../data/point_xs_MAIKo_test.npy")),
    axis=1)
cross_point_test = np.concatenate(
    (cross_point_test[:,0:1],
     cross_point_test[:,2:3],
     cross_point_test[:,3:4],
     cross_point_test[:,5:6]),
    axis=1)

batch_size = 16
epochs = 1000

##========== tensorboard setup ==========##
old_session = KTF.get_session()
session = tf.Session("")
KTF.set_session(session)
KTF.set_learning_phase(1)

##========== building model ==========##
anode_input = Input(shape=anode[0].shape)
x0 = MaxPooling2D(pool_size=(4,4))(anode_input)
x1 = Conv2D(filters=64,kernel_size=(3,3),
            padding="same",activation="relu")(x0)
x2 = Conv2D(filters=64,kernel_size=(3,3),
            padding="same",activation="relu")(x1)
x3 = MaxPooling2D(pool_size=(2,2))(x2)
x4 = Conv2D(filters=128,kernel_size=(3,3),
            padding="same",activation="relu")(x3)
x5 = Conv2D(filters=128,kernel_size=(3,3),
            padding="same",activation="relu")(x4)
x6 = MaxPooling2D(pool_size=(2,2))(x5)
x7 = Conv2D(filters=256,kernel_size=(3,3),
            padding="same",activation="relu")(x6)
x8 = Conv2D(filters=256,kernel_size=(3,3),
            padding="same",activation="relu")(x7)
x9 = Conv2D(filters=256,kernel_size=(3,3),
            padding="same",activation="relu")(x8)
x10 = MaxPooling2D(pool_size=(2,2))(x9)
z1 = Conv2D(filters=512,kernel_size=(3,3),
            padding="same",activation="relu")(x10)
z2 = Conv2D(filters=512,kernel_size=(3,3),
            padding="same",activation="relu")(z1)
z3 = Conv2D(filters=512,kernel_size=(1,1),
            padding="same",activation="relu")(z2)
z4 = MaxPooling2D(pool_size=(2,2))(z3)
z5 = Conv2D(filters=512,kernel_size=(3,3),
            padding="same",activation="relu")(z4)
z6 = Conv2D(filters=512,kernel_size=(3,3),
            padding="same",activation="relu")(z5)
z7 = Conv2D(filters=512,kernel_size=(1,1),
            padding="same",activation="relu")(z6)
z8 = MaxPooling2D(pool_size=(2,2))(z7)
z9 = Flatten()(z8)
z9 = Dense(4096,activation="sigmoid")(z9)
z10 = Dropout(0.5)(z9)
z11 = Dense(4096,activation="sigmoid")(z10)
z12 = Dropout(0.5)(z11)
z13 = Dense(4,activation="sigmoid")(z12)
z14 = Dropout(0.5)(z13)
output = Dense(4,activation="relu")(z14)

model = Model(inputs=anode_input,outputs=output)

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="mse",
              optimizer=sgd)

##========== fitting ==========##
csvlogger = CSVLogger("detector_vgg.csv")
board = TensorBoard(log_dir="log",histogram_freq=1)
model.fit(anode,
          cross_point,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=[anode_test,cross_point_test],
          callbacks=[csvlogger,board])
model.save("detector_vgg.h5")
KTF.set_session(old_session)
