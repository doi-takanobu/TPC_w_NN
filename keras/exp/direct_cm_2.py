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


if TENSOR_BOARD==1:
    old_session = KTF.get_session()
    session = tf.Session("")
    KTF.set_session(session)
    KTF.set_learning_phase(1)

## model build
Input = Input(shape=shape)
x1_1 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",data_format="channels_first")(Input)
#x1_2 = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x1_1)
x2 = MaxPooling2D(pool_size=(4,4))(x1_1)
x3_1 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu")(x2)
#x3_2 = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu")(x3_1)
x4 = MaxPooling2D(pool_size=(4,4))(x3_1)
x5 = Flatten()(x4)
x6 = Dense(128,activation="sigmoid")(x5)
y6 = Dense(128,activation="sigmoid")(x5)
x7 = Dropout(0.5)(x6)
y7 = Dropout(0.5)(y6)
x8 = Dense(16,activation="sigmoid")(x7)
y8 = Dense(16,activation="sigmoid")(y7)
x9 = Dropout(0.5)(x8)
y9 = Dropout(0.5)(y8)
Output_dx = Dense(1,activation="relu")(x9)
Output_thr = Dense(2,activation="relu")(y9)

model = Model(inputs=Input,outputs=[Output_dx,Output_thr])
#sgd = SGD(lr=0.01,momentum=0.9,decay=1.e-6,nesterov=True)
#model.compile(optimizer=sgd,loss="mse")
#plot_model(model,"direct_exp.png")
model.compile(optimizer="Adadelta",loss="logcosh")

csvlogger = CSVLogger("direct_exp_cm_2.csv")
cb = [csvlogger]
if TENSOR_BOARD==1:
    board = TensorBoard(log_dir="log_direct",histogram_freq=1)
    cb.append(board)
start = time.time()
model.fit(cell,[dx/10.,np.concatenate((thr,phi),axis=1)],
          validation_data=[cell_test,[dx_test/10.,np.concatenate((thr_test,phi_test),axis=1)]]
         ,epochs=300,batch_size=64,callbacks=cb)
end = time.time()
print("Fitting time: ",end-start,"s")

model.save("direct_exp_cm_2.h5")

print("begin prediction")
start = time.time()
pred_dx,pred_thr = model.predict(cell_test)
end = time.time()
np.savetxt("direct_exp_cm_2.dat",np.concatenate((pred_dx,pred_thr),axis=1),header="dx theta phi")
print("end prediction")
print("Prediction time: ",end-start,"s")
