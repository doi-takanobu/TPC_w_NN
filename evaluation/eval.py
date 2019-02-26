import sys
import numpy as np
import keras
import time

#--------- define functions to evaluate --------#
def acc(true,pred):
    correct = 0.
    population = 0.
    if len(true) != len(pred):
        exit
    for i in range(len(true)):
        population += 1.
        if true[i].argmax() == pred[i].argmax():
            correct += 1.
    return correct/population

def eff(true,pred):
    correct = 0.
    population = 0.
    if len(true) != len(pred):
        exit
    for i in range(len(true)):
        if true[i].argmax() == 0:
            population += 1.
            if pred[i].argmax() == 0:
                correct += 1.
    return correct/population

def pur(true,pred):
    correct = 0.
    population = 0.
    if len(true) != len(pred):
        exit
    for i in range(len(true)):
       if pred[i].argmax() == 0:
           population += 1.
           if true[i].argmax() == 0:
               correct += 1.
    return correct/population

def jac(true,pred):
    correct = 0.
    population = 0.
    if len(true) != len(pred):
        exit
    for i in range(len(true)):
        population += 1.
        if (true[i].argmax() == 1 and pred[i].argmax() ==1):
            population -= 1.
        if (true[i].argmax() == 0 and pred[i].argmax() == 0):
            correct += 1.
    return correct/population

def judge(true,pred):
    tt = 0
    tf = 0
    ft = 0
    ff = 0
    if len(true) != len(pred):
        exit
    for i in range(len(true)):
        if true[i].argmax() == 0 and pred[i].argmax() == 0:
            tt += 1
        if true[i].argmax() == 0 and pred[i].argmax() == 1:
            tf += 1
        if true[i].argmax() == 1 and pred[i].argmax() == 0:
            ft += 1
        if true[i].argmax() == 1 and pred[i].argmax() == 1:
            ff += 1
    return tt,tf,ft,ff

model = keras.models.load_model(sys.argv[1])
anode = np.load("../data/anode_uint8.npy")[2700:3000]
cathode = np.load("../data/cathode_uint8.npy")[2700:3000]
ans = np.load("../data/ans.npy")[2700:3000]
anode = anode.reshape((-1,256,1024,1))
cathode = cathode.reshape((-1,256,1024,1))

start = time.time()
pred = model.predict([anode,cathode])
end = time.time()

acc = acc(ans,pred)
eff = eff(ans,pred)
pur = pur(ans,pred)
jac = jac(ans,pred)
tt,tf,ft,ff = judge(ans,pred)

f = open(sys.argv[1].split("/")[-1].split(".")[0]+"_eval.dat","w")
f.write("prediction time: "+str(end-start)+"s\n")
f.write("accuracy: "+str(acc)+"\n")
f.write("efficiency: "+str(eff)+"\n")
f.write("purity: "+str(pur)+"\n")
f.write("jaccard: "+str(jac)+"\n")
f.write("\ntt: "+str(tt))
f.write("  tf: "+str(tf))
f.write("  ft: "+str(ft))
f.write("  ff: "+str(ff))
f.close()
