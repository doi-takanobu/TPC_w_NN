import numpy as np
import time
from keras.models import load_model

path = "../../data/"

cell = np.empty((0,2,1024,256))
point = np.empty((0,8))
for i in [234,235,236,237,238,239,241,242]:
    cell = np.append(cell,np.load(path+"track/run0"+str(i)+"_track.npy"),axis=0)
    point = np.append(point,np.load(path+"result/run0"+str(i)+"_result.npy")[:,5:13],axis=0)
    print(i,len(cell))

model = load_model("indirect_norm-7.h5")
start = time.time()
pred = model.predict([cell[:,0:1],cell[:,1:2]])
end = time.time()
print(end-start,"sec")
factor = np.array([256.,1024.,256.,1024.,256.,1024.,256.,1024.])
np.savetxt("indirect_test_ans-6.dat",point,header="avs avc aes aec cvs csc ces cec [pixel] (232--242)")
np.savetxt("indirect_test_pred-6.dat",pred*factor,header="avs avc aes aec cvs csc ces cec [pixel] (232--242)")
