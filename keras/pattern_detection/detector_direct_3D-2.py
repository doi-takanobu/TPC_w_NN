import keras
import numpy as np
import time
from keras.models import Model
from keras.layers import Input,Dense,Dropout
from keras.layers import MaxPooling2D,AveragePooling2D
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
shape = cell[0].shape

dx = np.sqrt((xs[:,0:1]-xv[:,0:1])*(xs[:,0:1]-xv[:,0:1])+(xs[:,1:2]-xv[:,1:2])*(xs[:,1:2]-xv[:,1:2])+(xs[:,2:3]-xv[:,2:3])*(xs[:,2:3]-xv[:,2:3]))
dx_test = np.sqrt((xs_test[:,0:1]-xv_test[:,0:1])*(xs_test[:,0:1]-xv_test[:,0:1])+(xs_test[:,1:2]-xv_test[:,1:2])*(xs_test[:,1:2]-xv_test[:,1:2])+(xs_test[:,2:3]-xv_test[:,2:3])*(xs_test[:,2:3]-xv_test[:,2:3]))
thr = np.arccos((xs[:,0:1]-xv[:,0:1])/dx)
thr_test = np.arccos((xs_test[:,0:1]-xv_test[:,0:1])/dx_test)
phi = np.arccos((xs[:,1:2]-xv[:,1:2])/(dx*np.sin(thr)))
phi_test = np.arccos((xs_test[:,1:2]-xv_test[:,1:2])/(dx_test*np.sin(thr_test)))
thr = np.degrees(thr)
thr_test = np.degrees(thr_test)
phi = np.degrees(phi)
phi_test = np.degrees(phi_test)

if TENSOR_BOARD==1:
    old_session = KTF.get_session()
    session = tf.Session("")
    KTF.set_session(session)
    KTF.set_learning_phase(1)

## model build
Input = Input(shape=shape)
x0 = AveragePooling2D(pool_size=(4,4))(Input)
x1 = Conv2D(filters=32,kernel_size=(8,8),padding="same",activation="relu")(x0)
y1 = Conv2D(filters=32,kernel_size=(8,8),padding="same",activation="relu")(x0)
x2 = MaxPooling2D(pool_size=(8,8))(x1)
y2 = MaxPooling2D(pool_size=(8,8))(y1)
x3 = Conv2D(filters=32,kernel_size=(8,8),padding="same",activation="relu")(x2)
y3 = Conv2D(filters=32,kernel_size=(8,8),padding="same",activation="relu")(y2)
x4 = MaxPooling2D(pool_size=(8,8))(x3)
y4 = MaxPooling2D(pool_size=(8,8))(y3)
x5 = Flatten()(x4)
y5 = Flatten()(y4)
x6 = Dense(256,activation="sigmoid")(x5)
y6 = Dense(256,activation="sigmoid")(y5)
x7 = Dropout(0.5)(x6)
y7 = Dropout(0.5)(y6)
Output_dx = Dense(1,activation="relu")(x7)
Output_thr = Dense(2,activation="relu")(y7)

model = Model(inputs=Input,outputs=[Output_dx,Output_thr])
#sgd = SGD(lr=0.01,momentum=0.9,decay=1.e-6,nesterov=True)
#model.compile(optimizer=sgd,loss="mse")
model.compile(optimizer="Adadelta",loss="mse")
#plot_model(model,"direct-1_2.png")

csvlogger = CSVLogger("direct_3D.csv")
cb = [csvlogger]
if TENSOR_BOARD==1:
    board = TensorBoard(log_dir="log_direct",histogram_freq=1)
    cb.append(board)
start = time.time()
model.fit(cell,[dx,np.concatenate((thr,phi),axis=1)],
          validation_data=[cell_test,[dx_test,np.concatenate((thr_test,phi_test),axis=1)]]
         ,epochs=1500,batch_size=128,callbacks=cb)
end = time.time()
print("Fitting time: ",end-start,"s")

model.save("direct_3D.h5")

print("begin prediction")
pred_dx,pred_thr = model.predict(cell_test)
np.savetxt("direct_3D.dat",np.concatenate((pred_dx,pred_thr),axis=1))
print("end prediction")
