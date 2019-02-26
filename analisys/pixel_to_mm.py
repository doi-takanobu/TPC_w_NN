import numpy as np
def pixel_to_mm(data,factor):
    data = np.concatenate((data[:,0:1]*0.4,data[:,1:2]*factor,data[:,2:3]*0.4,data[:,3:4]*factor,data[:,4:5]*0.4,data[:,5:6]*factor,data[:,6:7]*0.4,data[:,7:8]*factor),axis=1)
    return data
def dx_theta_phi(data):
    dx = np.sqrt((data[:,0:1]-data[:,2:3])*(data[:,0:1]-data[:,2:3])+
                 (data[:,4:5]-data[:,6:7])*(data[:,4:5]-data[:,6:7])+
                 (((data[:,1:2]+data[:,5:6])/2.)-((data[:,3:4]+data[:,7:8])/2.))*(((data[:,1:2]+data[:,5:6])/2.)-((data[:,3:4]+data[:,7:8])/2.)))
    theta = (data[:,0:1]-data[:,2:3])/dx
    theta = np.degrees(np.arccos(theta))
    phi = (data[:,4:5]-data[:,6:7])/((data[:,3:4]+data[:,7:8])/2.)
    phi = np.degrees(np.arctan(phi))
    return dx,theta,phi
