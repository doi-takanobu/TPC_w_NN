import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense,Dropout,concatenate
from keras.callbacks import CSVLogger#,TensorBoard
#from keras.utils import plot_model
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf

path = "../../data/"

cell = np.empty((0,2,1024,256))
result = np.empty((0,13))
for i in [178,179,182,183,184]:
    cell = np.append(cell,np.load(path+"track/run0"+str(i)+"_track.npy"),axis=0)
    result = np.append(result,np.load(path+"result/run0"+str(i)+"_result.npy"),axis=0)
    print(i,len(cell))
point = result[:,5:13]
#point_a = result[:,5:9]
#point_c = result[:,9:13]
cell_test = np.empty((0,2,1024,256))
result_test = np.empty((0,13))
for i in [190,191]:
    cell_test = np.append(cell_test,np.load(path+"track/run0"+str(i)+"_track.npy"),axis=0)
    result_test = np.append(result_test,np.load(path+"result/run0"+str(i)+"_result.npy"),axis=0)
    print(i,len(cell_test))
point_test = result_test[:,5:13]
#point_a_test = result_test[:,5:9]
#point_c_test = resutl_test[:,9:13]
del result,result_test
shape = cell[0][0:1].shape

#old_session = KTF.get_session()
#session = tf.Session('')
#KTF.set_session(session)
#KTF.set_learning_phase(1)

Input_a = Input(shape=shape)
Input_c = Input(shape=shape)
x = Conv2D(filters=16,kernel_size=3,padding="same",
           activation="relu",data_format="channels_first")(Input_a)
y = Conv2D(filters=16,kernel_size=3,padding="same",
           activation="relu",data_format="channels_first")(Input_c)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y)
x = Conv2D(filters=64,kernel_size=3,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=64,kernel_size=3,padding="same",
           activation="relu",data_format="channels_first")(y)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y)
x = Conv2D(filters=256,kernel_size=3,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=256,kernel_size=3,padding="same",
           activation="relu",data_format="channels_first")(y)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y)
x = Flatten()(x)
y = Flatten()(y)
x = Dense(256,activation="sigmoid")(x)
y = Dense(256,activation="sigmoid")(y)
x = Dropout(0.5)(x)
y = Dropout(0.5)(y)
x = Dense(16,activation="sigmoid")(x)
y = Dense(16,activation="sigmoid")(y)
x = Dropout(0.5)(x)
y = Dropout(0,5)(y)
z = concatenate([x,y])
Output = Dense(8)(z)

model = Model(inputs=[Input_a,Input_c],outputs=Output)
model.compile(loss="mse",optimizer="adam")
csvlogger = CSVLogger("indirect_norm.csv")
#board = TensorBoard(log_dir="./log/",histogram_freq=1)

factor = [256.,1024.,256.,1024.,256.,1024.,256.,1024.]
factor = np.array(factor)

start = time.time()
model.fit([cell[:,0:1],cell[:,1:2]],point/factor*2-1,epochs=500,batch_size=64,
          validation_data=[[cell_test[:,0:1],cell_test[:,1:2]],point_test/factor*2-1],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
model.summary()
model.save("indirect_norm.h5")

pred = model.predict([cell_test[:,0:1],cell_test[:,1:2]])
np.savetxt("indirect_norm.dat",(pred+1)*factor/2,header="avs avc aes aec cvs cvc ces cec [pixel]")

#KTF.set_session(old_session)
