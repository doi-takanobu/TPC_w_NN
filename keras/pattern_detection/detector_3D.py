import keras
import numpy as np
import time
from keras.models import Model
from keras.layers import Input,Dense,Dropout
from keras.layers import MaxPooling2D,SeparableConv2D
from keras.layers import Conv2D,Flatten
from keras.optimizers import SGD
from keras.callbacks import CSVLogger,TensorBoard
from keras.utils import plot_model
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf

TENSOR_BOARD = 0 # 0:OFF 1:ON
path = "../../data/"

cell = np.load(path+"cell_a_MAIKo_3D.npy")
cell = cell.reshape((-1,cell.shape[1],cell.shape[2],1))
cell_test = np.load(path+"cell_a_MAIKo_3D_test.npy")
cell_test = cell_test.reshape((-1,cell_test.shape[1],cell_test.shape[2],1))
xv = np.load(path+"point_xv_MAIKo_3D.npy")
xs = np.load(path+"point_xs_MAIKo_3D.npy")
xv_test = np.load(path+"point_xv_MAIKo_3D_test.npy")
xs_test = np.load(path+"point_xs_MAIKo_3D_test.npy")
shape = cell[0,:,:,0:1].shape
print(shape)

if TENSOR_BOARD==1:
    old_session = KTF.get_session()
    session = tf.Session("")
    KTF.set_session(session)
    KTF.set_learning_phase(1)

## model build
Input_a = Input(shape=shape)
Input_c = Input(shape=shape)
x1 = Conv2D(filters=8,kernel_size=3,padding="same",activation="relu")(Input_a)
x2 = MaxPooling2D(pool_size=(4,4))(x1)
x3 = Conv2D(filters=32,kernel_size=3,padding="same",activation="relu")(x2)
x4 = MaxPooling2D(pool_size=(4,4))(x3)
x5 = Conv2D(filters=128,kernel_size=3,padding="same",activation="relu")(x4)
x6 = MaxPooling2D(pool_size=(4,4))(x5)
x7 = Flatten()(x6)
x8 = Dense(64,activation="sigmoid")(x7)
x9 = Dropout(0.5)(x8)
Output = Dense(6,activation="relu")(x9)

model = Model(inputs=Input,outputs=Output)
#sgd = SGD(lr=0.01,momentum=0.9,decay=1.e-6,nesterov=True)
#model.compile(optimizer=sgd,loss="mse")
model.compile(optimizer="Adadelta",loss="mse")
#plot_model(model,"direct-1_2.png")

csvlogger = CSVLogger("3D.csv")
cb = [csvlogger]
if TENSOR_BOARD==1:
    board = TensorBoard(log_dir="log_direct",histogram_freq=1)
    cb.append(board)
start = time.time()
model.fit(cell,np.concatenate((xv,xs),axis=1),
          validation_data=[cell_test,np.concatenate((xv_test,xs_test),axis=1)]
         ,epochs=200,batch_size=64,callbacks=cb)
end = time.time()
print("Fitting time: ",end-start,"s")

model.save("3D.h5")

print("begin prediction")
point = model.predict(cell_test)
np.savetxt("3D.dat",point,header="xv1 xv2 xv3 xs1 xs2 xs3")
print("end prediction")
