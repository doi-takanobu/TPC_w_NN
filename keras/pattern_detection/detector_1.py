import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense,Dropout
from keras.callbacks import CSVLogger,TensorBoard
from keras.utils import plot_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

path = "../../data/"

cell = np.load(path+"cell_a_MAIKo.npy")
cell = cell.reshape((-1,
                     cell.shape[1],
                     cell.shape[2],
                     1))
cell_test = np.load(path+"cell_a_MAIKo_test.npy")
cell_test = cell_test.reshape((-1,
                               cell_test.shape[1],
                               cell_test.shape[2],
                               1))
cross_point = np.concatenate(
    (np.load(path+"point_xv_MAIKo.npy")[:,0:1],
     np.load(path+"point_xv_MAIKo.npy")[:,2:3],
     np.load(path+"point_xs_MAIKo.npy")[:,0:1],
     np.load(path+"point_xs_MAIKo.npy")[:,2:3]),
                             axis=1)
cross_point_test = np.concatenate(
    (np.load(path+"point_xv_MAIKo_test.npy")[:,0:1],
     np.load(path+"point_xv_MAIKo_test.npy")[:,2:3],
     np.load(path+"point_xs_MAIKo_test.npy")[:,0:1],
     np.load(path+"point_xs_MAIKo_test.npy")[:,2:3]),
                                 axis=1)
shape = cell[0].shape

old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

Input = Input(shape=shape)
x = MaxPooling2D(pool_size=(4,4))(Input)
x = Conv2D(filters=64,kernel_size=3,padding="same",
           activation="relu")(x)
x = MaxPooling2D(pool_size=(4,4))(x)
x = Conv2D(filters=256,kernel_size=3,padding="same",
           activation="relu")(x)
x = Flatten()(x)
x = Dense(512,activation="sigmoid")(x)
x = Dropout(0.5)(x)
x = Dense(64,activation="sigmoid")(x)
x = Dropout(0.5)(x)
Output = Dense(4,activation="relu")(x)

model = Model(inputs=Input,outputs=Output)
model.compile(loss="mse",optimizer="SGD")
csvlogger = CSVLogger("detector_1.csv")
#board = TensorBoard(log_dir="./log/",histogram_freq=1)

model.summary()
start = time.time()
model.fit(cell,cross_point,epochs=1000,batch_size=50,
          validation_data=[cell_test,cross_point_test],
          callbacks=[csvlogger])
end = time.time()
model.save("detector1.h5")
print("Learning time is {} second".format(end-start))


KTF.set_session(old_session)
