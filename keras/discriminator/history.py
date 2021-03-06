import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fnormal = "discri_0.csv"
fvgg = "discri_vgg.csv"
normal = np.loadtxt(fnormal,skiprows=1,delimiter=",")
vgg = np.loadtxt(fvgg,skiprows=1,delimiter=",")
print(normal.shape)
print(vgg.shape)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.xmargin"] = "0"
plt.rcParams["axes.ymargin"] = "0"
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)
ax.plot(normal[:,0],normal[:,1],label="train data (simple CNN)")
ax.plot(normal[:,0],normal[:,3],label="test data (simple CNN)")
ax.plot(vgg[:,0],vgg[:,1],label="train data (VGG-16)")
ax.plot(vgg[:,0],vgg[:,3],label="test data (VGG-16)")
ax.set_title("")
ax.set_xlabel("epochs")
ax.set_ylabel("accurary (%)")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
#ax.set_xlim(0,201)
ax.set_ylim(0,1.1)
ax.legend(loc="lower right")
xtick = np.linspace(0,200,5)
xlocs = np.linspace(0,200,5)
ax.set_xticks(xlocs)
ax.set_xticklabels(xtick)
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
fig.show()

fig.savefig("event_selection_history.eps",bbox="tight")
#fig.savefig(fname.split(".")[0]+"_history.eps",bbox_inches="tight")
