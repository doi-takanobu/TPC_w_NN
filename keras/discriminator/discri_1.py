import keras
from keras.models import Model
from keras.layers import Dense,Dropout,Input
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras.callbacks import CSVLogger,TensorBoard
from keras.utils import plot_model
import numpy as np
import time

##========== data loading ==========##
anode = np.load("../../data/anode_uint8.npy")
cathode = np.load("../../data/cathode_uint8.npy")
anode = anode.reshape((-1,256,1024,1))
cathode = cathode.reshape((-1,256,1024,1))
ans = np.load("../../data/ans.npy")

batch_size = 64
epochs = 200
sample_size = 3000

#anode = anode[:sample_size]
#cathode = cathode[:sample_size]
#ans = ans[:sample_size]

##========== model building ==========##
anode_input = Input(shape=(256,1024,1))
cathode_input = Input(shape=(256,1024,1))
x = MaxPooling2D(pool_size=(4,4))(anode_input)
y = MaxPooling2D(pool_size=(4,4))(cathode_input)
x = Conv2D(filters=40,kernel_size=(16,16),
           padding="same",activation="relu")(x)
y = Conv2D(filters=40,kernel_size=(16,16),
           padding="same",activation="relu")(y)
x = MaxPooling2D(pool_size=(4,4))(x)
y = MaxPooling2D(pool_size=(4,4))(y)
x = Conv2D(filters=40,kernel_size=(8,8),
           padding="same",activation="relu")(x)
y = Conv2D(filters=40,kernel_size=(8,8),
           padding="same",activation="relu")(y)
x = MaxPooling2D(pool_size=(4,4))(x)
y = MaxPooling2D(pool_size=(4,4))(y)
z = keras.layers.concatenate([x,y])
z = Conv2D(filters=30,kernel_size=(4,4),
           padding="same",activation="relu")(z)
z = Flatten()(z)
z = Dense(256,activation="sigmoid")(z)
z = Dropout(0.3)(z)
output = Dense(2,activation="softmax")(z)

model = Model(inputs=[anode_input,cathode_input],outputs=output)

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])
#plot_model(model,"discri_1.png")

##========== fitting ==========##
csvlogger = CSVLogger("discri_1.csv")
#board = TensorBoard(log_dir="log")
start = time.time()
model.fit([anode,cathode],
          ans,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1,
          callbacks=[csvlogger])
end = time.time()
print("Fitting time is {} second.".format(end-start))
model.save("discri_1.h5")

start = time.time()
pred = model.predict([anode[int(0.9*sampe_size):],cathode[int(0.9*sample_size):]])
end = time.time()
np.savetxt("discri_1.dat",pred)
print("Predicting time is {} sectond.".format(end-start))
#score = model.evaluate([anode[int(0.9*len(anode)):],
#                        cathode[int(0.9*len(cathode)):]],
#                       ans[int(0.9*len(ans)):],
#                       batch_size=batch_size)
#print(score)
