import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fname = "indirect_norm-7.csv"
history = np.loadtxt(fname,skiprows=1,delimiter=",")
print(history.shape)

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
ax.plot(history[:,0],history[:,1],label="train data")
ax.plot(history[:,0],history[:,2],label="test data")
ax.set_title("")
ax.set_xlabel("epochs")
ax.set_ylabel("mean square error (a.u.)")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.set_yscale("log")
#ax.set_xlim(0,201)
#ax.set_ylim(0,1.1)
ax.legend(loc="upper right")
#xtick = np.linspace(0,200,5)
#xlocs = np.linspace(0,200,5)
#ax.set_xticks(xlocs)
#ax.set_xticklabels(xtick)
#ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
#plt.show()
fig.show()
fig.savefig(fname.split(".")[0]+"_history.eps",bbox_inches="tight")
