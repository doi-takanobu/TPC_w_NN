import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("discri_1.h5")

anode = np.load("../../data/anode_uint8.npy")
cathode = np.load("../../data/cathode_uint8.npy")
ans = np.load("../../data/ans.npy")
pred = model.predict([anode.reshape((-1,256,1024,1)),cathode.reshape((-1,256,1024,1))])

for i in range(10):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.imshow(anode[i+3100].reshape((256,1024)),cmap=plt.cm.binary)
    ax2.imshow(cathode[i+3100].reshape((256,1024)),cmap=plt.cm.binary)
    fig.suptitle("{0},{1}".format(pred[i+3100],ans[i+3100]))
    fig.show()
    input("Enter")
    plt.close(fig)
