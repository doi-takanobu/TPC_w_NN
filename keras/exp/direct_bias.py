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
shape = cell[0].shape
print(shape)


if TENSOR_BOARD==1:
    old_session = KTF.get_session()
    session = tf.Session("")
    KTF.set_session(session)
    KTF.set_learning_phase(1)

## model build
Input = Input(shape=shape)
x1_1 = Conv2D(filters=8,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(Input)
#x1_1 = SeparableConv2D()(Input)
x1_2 = Conv2D(filters=8,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x1_1)
y1_1 = Conv2D(filters=8,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(Input)
y1_2 = Conv2D(filters=8,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y1_1)
x2 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x1_2)
y2 = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y1_2)
#x3_1 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x2)
#x3_2 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x3_1)
#y3_1 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y2)
#y3_2 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y3_1)
#x4 = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x3_2)
#y4 = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y3_2)
#x5_1 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x4)
#x5_2 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x5_1)
#x5_3 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x5_2)
#y5_1 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x4)
#y5_2 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y5_1)
#y5_3 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y5_2)
#x6 = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x5_3)
#y6 = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y5_3)
#x7_1 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x6)
#x7_2 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(x7_1)
#x7_3 = Conv2D(filters=64,kernel_size=(1,1),padding="same",activation="relu",data_format="channels_first")(x7_2)
#y7_1 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y6)
#y7_2 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(y7_1)
#y7_3 = Conv2D(filters=64,kernel_size=(1,1),padding="same",activation="relu",data_format="channels_first")(y7_2)
x8 = Flatten()(x2)
y8 = Flatten()(y2)
x9 = Dense(256,activation="sigmoid")(x8)
y9 = Dense(256,activation="sigmoid")(y8)
x10 = Dropout(0.5)(x9)
y10 = Dropout(0.5)(y9)
x11 = Dense(16,activation="sigmoid")(x10)
y11 = Dense(16,activation="sigmoid")(y10)
x12 = Dropout(0.5)(x11)
y12 = Dropout(0.5)(y11)
Output_dx = Dense(1)(x12)
Output_thr = Dense(1)(x12)

model = Model(inputs=Input,outputs=[Output_dx,Output_thr])
#sgd = SGD(lr=0.1,momentum=0.9,decay=1.e-6,nesterov=True)
#model.compile(optimizer=sgd,loss="logcosh")
#plot_model(model,"direct_exp.png")
model.compile(optimizer="Adagrad",loss="mse")

csvlogger = CSVLogger("direct_exp_bias.csv")
cb = [csvlogger]
if TENSOR_BOARD==1:
    board = TensorBoard(log_dir="log_direct",histogram_freq=1)
    cb.append(board)
start = time.time()
model.fit(cell,[dx-45,thr-88],
          validation_data=[cell_test,[dx_test-45,thr_test-88]]
         ,epochs=300,batch_size=64,callbacks=cb)
end = time.time()
print("Fitting time: ",end-start,"s")

model.save("direct_exp_bias.h5")

print("begin prediction")
start = time.time()
pred_dx,pred_thr = model.predict(cell_test)
pred_dx = pred_dx+45
pred_thr = pred_thr+88
end = time.time()
np.savetxt("direct_exp_bias.dat",np.concatenate((pred_dx,pred_thr),axis=1),header="dx theta")
print("end prediction")
print("Prediction time: ",end-start,"s")
