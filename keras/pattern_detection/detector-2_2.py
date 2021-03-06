import numpy as np
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Flatten
from keras.layers import Dense,Dropout
from keras.callbacks import CSVLogger,TensorBoard
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf
import time

TENSOR_BOARD = 1 # 0:OFF 1:ON <-- output tensorboard
path = "../../data/"

cell = np.load(path+"cell_a_MAIKo.npy")
cell = cell.reshape((-1,cell.shape[1],cell.shape[2],1))
print(cell.shape)
cell_test = np.load(path+"cell_a_MAIKo_test.npy")
cell_test = cell_test.reshape((-1,cell_test.shape[1],cell_test.shape[2],1))
print(cell_test.shape)
xv = np.load(path+"point_xv_MAIKo.npy")
xs = np.load(path+"point_xs_MAIKo.npy")
cross_point = np.concatenate((xv[:,0:1],xv[:,2:3],xs[:,0:1],xs[:,2:3]),axis=1)
print(cross_point.shape)
xv_test = np.load(path+"point_xv_MAIKo_test.npy")
xs_test = np.load(path+"point_xs_MAIKo_test.npy")
cross_point_test = np.concatenate((xv_test[:,0:1],xv_test[:,2:3],xs_test[:,0:1],xs_test[:,2:3]),axis=1)
print(cross_point_test.shape)
shape = cell[0].shape

if TENSOR_BOARD==1:
    old_session = KTF.get_session()
    session = tf.Session("")
    KTF.set_session(session)
    KTF.set_learning_phase(1)

input = Input(shape=shape)
x = AveragePooling2D(pool_size=(2,8))(input)
x = Conv2D(filters=32,kernel_size=(16,16),padding="same",activation="relu")(x)
x = MaxPooling2D(pool_size=(4,4))(x)
x = Conv2D(filters=32,kernel_size=(8,8),padding="same",activation="relu")(x)
x = MaxPooling2D(pool_size=(4,4))(x)
#x = Conv2D(filters=32,kernel_size=(8,8))(x)
x= Flatten()(x)
x = Dense(256,activation="sigmoid")(x)
x = Dropout(0.5)(x)
output = Dense(4,activation="relu")(x)

model = Model(inputs=input,outputs=output)
model.compile(loss="mse",optimizer="SGD")
csvlogger = CSVLogger("history_detect-2_2.csv")
cb = [csvlogger]

if TENSOR_BOARD==1:
    board = TensorBoard(log_dir="log_indirect",histogram_freq=1)
    cb.append(board)
    
model.fit(cell,cross_point/10.,epochs=300,batch_size=50,verbose=2,
          validation_data=[cell_test,cross_point_test/10.],callbacks=cb)
model.save("model_detect-2_2.h5")
if TENSOR_BOARD==1:
    KTF.set_session(old_session)
