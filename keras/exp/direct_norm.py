import keras
import numpy as np
import time
from keras.models import Model
from keras.layers import Input,Dense,Dropout
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.layers import Conv2D,Flatten,concatenate
from keras.optimizers import SGD
from keras.callbacks import CSVLogger,TensorBoard
from keras.utils import plot_model
from keras.backend import tensorflow_backend as KTF
import tensorflow as tf

TENSOR_BOARD = 0 # 0:OFF 1:ON
path = "../../data/"

#178--183 for fit
#190--191 for test
cell = np.empty((0,2,1024,256))
result = np.empty((0,13))
for i in [178,179,182,183,184]:   
    cell = np.append(cell,np.load(path+"track/run0"+str(i)+"_track.npy"),axis=0)
    result = np.append(result,np.load(path+"result/run0"+str(i)+"_result.npy"),axis=0)
    print(i,len(cell))
dx = result[:,2:3]
thr = result[:,0:1]
phi = result[:,1:2]
cell_test = np.empty((0,2,1024,256))
result_test = np.empty((0,13))
for i in [190,191]:
    cell_test = np.append(cell_test,np.load(path+"track/run0"+str(i)+"_track.npy"),axis=0)
    result_test = np.append(result_test,np.load(path+"result/run0"+str(i)+"_result.npy"),axis=0)
    print(i,len(cell_test))
dx_test = result_test[:,2:3]
thr_test = result_test[:,0:1]
phi_test = result_test[:,1:2]
del result,result_test
shape = cell[0][0:1].shape


if TENSOR_BOARD==1:
    old_session = KTF.get_session()
    session = tf.Session("")
    KTF.set_session(session)
    KTF.set_learning_phase(1)

## model build
Input_a = Input(shape=shape)
Input_c = Input(shape=shape)
x1 = Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(Input_a)
y1 = Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(Input_c)
x2 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x1)
y2 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y1)
x3 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x2)
y3 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y2)
x4 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x3)
y4 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y3)
x5 = Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x4)
y5 = Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y4)
x6 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x5)
y6 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y5)
x7 = Flatten()(x6)
y7 = Flatten()(y6)
x8 = Dense(256,activation="sigmoid")(x7)
y8 = Dense(256,activation="sigmoid")(y7)
x9 = Dropout(0.5)(x8)
y9 = Dropout(0.5)(y8)
x10 = concatenate([x9,y9],axis=1)
x11 = Dense(32,activation="sigmoid")(x10)
x12 = Dropout(0.5)(x11)
Output = Dense(3,activation="relu")(x12)

model = Model(inputs=[Input_a,Input_c],outputs=Output)
#sgd = SGD(lr=0.01,momentum=0.9,decay=1.e-6,nesterov=True)
#model.compile(optimizer=sgd,loss="mse")
#plot_model(model,"direct_exp.png")
model.compile(optimizer="Adadelta",loss="mse")

csvlogger = CSVLogger("direct_norm.csv")
cb = [csvlogger]
if TENSOR_BOARD==1:
    board = TensorBoard(log_dir="log_direct",histogram_freq=1)
    cb.append(board)
factor = np.array([100.,90.,180.])
start = time.time()
model.fit([cell[:,0:1],cell[:,1:2]],np.concatenate((dx,thr,phi),axis=1)/factor,
          validation_data=[[cell_test[:,0:1],cell_test[:,1:2]],np.concatenate((dx_test,thr_test,phi_test),axis=1)/factor]
         ,epochs=500,batch_size=64,callbacks=cb)
end = time.time()
print("Fitting time: ",end-start,"s")

model.save("direct_norm.h5")

print("begin prediction")
start = time.time()
pred = model.predict([cell_test[:,0:1],cell_test[:,1:2]])
end = time.time()
np.savetxt("direct_exp.dat",(pred)*factor,header="dx theta phi")
print("end prediction")
print("Prediction time: ",end-start,"s")
