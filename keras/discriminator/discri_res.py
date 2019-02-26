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

epochs = 600
batch_size = 128
lr = 0.01
decay = 1e-6
momentum = 0.9

##========== data loading ==========##
anode = np.load("../../data/anode_uint8.npy")
anode = anode.reshape((-1,256,1024,1))
cathode = np.load("../../data/cathode_uint8.npy")
cathode = cathode.reshape((-1,256,1024,1))
ans = np.load("../../data/ans.npy")

##========== tensorboard setup ==========##
old_session = KTF.get_session()
session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

##========== model building ==========##
anode_input = Input(shape=anode[0].shape)
cathode_input = Input(shape=cathode[0].shape)
x0 = MaxPooling2D(pool_size=(4,4))(anode_input)
y0 = MaxPooling2D(pool_size=(4,4))(cathode_input)
x1 = Conv2D(filters=32,kernel_size=(16,16),
            padding="same",activation="relu")(x0)
y1 = Conv2D(filters=32,kernel_size=(16,16),
            padding="same",activation="relu")(y0)
x2 = MaxPooling2D(pool_size=(2,2))(x1)
y2 = MaxPooling2D(pool_size=(2,2))(y1)
x3 = Conv2D(filters=32,kernel_size=(8,8),
            padding="same",activation="relu")(x2)
y3 = Conv2D(filters=32,kernel_size=(8,8),
            padding="same",activation="relu")(y2)
x4 = MaxPooling2D(pool_size=(2,2))(x3)
y4 = MaxPooling2D(pool_size=(2,2))(y3)
z0 = concatenate([x4,y4])
z1 = Conv2D(filters=32,kernel_size=(4,4),
            padding="same",activation="relu")(z0)
z2 = MaxPooling2D(pool_size=(4,4))(z1)
x6 = Flatten()(x2)
y6 = Flatten()(y2)
x7 = Dense(128,activation="sigmoid")(x6)
y7 = Dense(128,activation="sigmoid")(y6)
x8 = Dropout(0.5)(x7)
y8 = Dropout(0.5)(y7)
x9 = Flatten()(x4)
y9 = Flatten()(y4)
x10 = concatenate([x9,x8])
y10 = concatenate([y9,y8])
x11 = Dense(128,activation="sigmoid")(x10)
y11 = Dense(128,activation="sigmoid")(y10)
x12 = Dropout(0.5)(x11)
y12 = Dropout(0.5)(y11)
z3 = Flatten()(z2)
z4 = concatenate([x12,y12,z3])
z5 = Dense(512,activation="sigmoid")(z4)
z6 = Dropout(0.5)(z5)
output = Dense(2,activation="softmax")(z6)

model = Model(inputs=[anode_input,cathode_input],outputs=output)
sgd = SGD(lr=lr,decay=decay,momentum=momentum,nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])
plot_model(model,"discri_res.png")

##========== fitting ==========##
csvlogger = CSVLogger("discri_res.csv")
board = TensorBoard(log_dir="log/",histogram_freq=1)
model.fit([anode,cathode],
          ans,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1,
          callbacks=[csvlogger,board])
model.save("discri_res.h5")

KTF.set_session(old_session)
