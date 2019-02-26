import numpy as np
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import Dense,Dropout,Flatten,concatenate
from keras.optimizers import SGD
from keras.callbacks import CSVLogger,TensorBoard
from keras.utils import plot_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

epochs = 1000
batch_size = 16
lr = 0.01
decay = 1e-6
momentum = 0.9

##========== data loading ==========##
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

##========== tensorboard setup ==========##
old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

##========== model building ==========##
anode_input = Input(shape=anode[0].shape)
x0 = MaxPooling2D(pool_size=(4,4))(anode_input)
x1 = Conv2D(filters=32,kernel_size=(16,16),
            padding="same",activation="relu")(x0)
x2 = MaxPooling2D(pool_size=(2,2))(x1)
x3 = Conv2D(filters=32,kernel_size=(8,8),
            padding="same",activation="relu")(x2)
x4 = MaxPooling2D(pool_size=(2,2))(x3)
z1 = Conv2D(filters=32,kernel_size=(4,4),
            padding="same",activation="relu")(x4)
z2 = MaxPooling2D(pool_size=(4,4))(z1)
x6 = Flatten()(x2)
x7 = Dense(128,activation="sigmoid")(x6)
x8 = Dropout(0.5)(x7)
x9 = Flatten()(x4)
x10 = concatenate([x9,x8])
x11 = Dense(128,activation="sigmoid")(x10)
x12 = Dropout(0.5)(x11)
z3 = Flatten()(z2)
z4 = concatenate([x12,z3])
z5 = Dense(512,activation="sigmoid")(z4)
z6 = Dropout(0.5)(z5)
output = Dense(4,activation="relu")(z6)

model = Model(inputs=anode_input,outputs=output)
sgd = SGD(lr=lr,decay=decay,momentum=momentum,nesterov=True)
model.compile(loss="mse",
              optimizer=sgd)

##========== fitting ==========##
csvlogger = CSVLogger("detector_res.csv")
board = TensorBoard(log_dir="log/",histogram_freq=1)
model.fit(anode,
          cross_point,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=[anode_test,cross_point_test],
          callbacks=[csvlogger,board])
model.save("detector_res.h5")

KTF.set_session(old_session)
