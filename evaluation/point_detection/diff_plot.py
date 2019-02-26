import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from mpl_toolkits.mplot3d import Axes3D
#from keras.models import load_model

path = "/hdd1/work1/neural/"
#factor = np.array([256.,1024.,256.,1024.,256.,1024.,256.,1024.])
#ans1 = np.empty((0,8))
#ans2 = np.empty((0,5))
#pred = np.empty((0,8))
#model = load_model(path+"/keras/exp/indirect_norm-7.h5")
#
#p = glob.glob(path+"data/result/*.npy")
#filename = []
#for i in range(len(p)):
#    temp = p[i].split("/")[-1].split("_")[0]
#    if (temp == "run0178" or
#        temp == "run0179" or
#        temp == "run0180" or
#        temp == "run0181" or
#        temp == "run0182" or
#        temp == "run0183" or
#        temp == "run0184" or
#        temp == "run0185" or
#        temp == "run0186" or
#        temp == "run0187" or
#        temp == "run0188" or
#        temp == "run0189"):
#        continue
#    print(temp)
#    filename.append(temp)
#
#start = time.time()
#for i in range(len(filename)):
#    cell = np.load(path+"data/track/"+filename[i]+"_track.npy")
#    ans = np.load(path+"data/result/"+filename[i]+"_result.npy")
#    ans1 = np.append(ans1,ans[:,5:13],axis=0)
#    ans2 = np.append(ans2,ans[:,:5],axis=0)
#    temp = model.predict([cell[:,0:1],cell[:,1:2]])
#    pred = np.append(pred,temp*factor,axis=0)
#    print(filename[i],len(pred))
#end = time.time()
#
#print((end-start)/60.," min.")
#np.savetxt("ans_1_all.dat",ans1,header="avs avc aes aec cvs csc ces cec [pixel]")
#np.savetxt("ans_2_all.dat",ans2,header="dx theta phi [degree]")
#np.savetxt("pred_all.dat",pred,header="avs avc aes aec cvs csc ces cec [pixel]")

ans = np.loadtxt("ans_1_all.dat")
pred = np.loadtxt("pred_all.dat")

diff = ans-pred
diff = diff[np.all(diff<1e8,axis=1),:]
factor = np.array([.4,.174,.4,.174,.4,.174,.4,.174])
diff = diff*factor

diff_v_x = diff[:,0]
diff_v_y = diff[:,4]
diff_v_z = (diff[:,1]+diff[:,5])/2.
diff_e_x = diff[:,2]
diff_e_y = diff[:,6]
diff_e_z = (diff[:,3]+diff[:,7])/2.

np.savetxt("diff_v_all.dat",np.concatenate((diff_v_x.reshape((-1,1)),diff_v_y.reshape((-1,1)),diff_v_z.reshape((-1,1))),axis=1),header="x y z [mm]")
np.savetxt("diff_e_all.dat",np.concatenate((diff_e_x.reshape((-1,1)),diff_e_y.reshape((-1,1)),diff_e_z.reshape((-1,1))),axis=1),header="x y z [mm]")

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")

#ax.set_xlim(-1,1)
#ax.set_ylim(-1,1)
#ax.set_zlim(-1,1)

ax.plot(diff_v_x,diff_v_y,diff_v_z,"ro",ms=4,mew=0.5)
ax.plot(diff_e_x,diff_e_y,diff_e_z,"bo",ms=4,mew=0.5)

v = np.sqrt(diff_v_x*diff_v_x+diff_v_y*diff_v_y+diff_v_z*diff_v_z)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.hist(v,bins=100,range=(0,20))
#ax2.hist(v,bins=100,range=(v.min(),v.max()))

e = np.sqrt(diff_e_x*diff_e_x+diff_e_y*diff_e_y+diff_e_z*diff_e_z)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.hist(e,bins=100,range=(0,20))
#ax3.hist(e,bins=100,range=(e.min(),e.max()))

plt.show()
