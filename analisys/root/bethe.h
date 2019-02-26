/** bethe.h ************/
/* 15/10/11 Created by T. Kawabata */

#include <math.h>

#ifndef _BETHE
#define _BEHTE 1
#define FAC 0.1535       /* 比例定数 MeVcm^2/g */
#define MASSE  0.51099906   /* 電子の静止質量 MeV */
#define MEGA 1000000               
#define BETH_BIN 10000 /* Number of steps in the energy loss calculation */
#define BETH_ESTEP 10 /* Precision (eV) in the enegy loss calcuation. */ 

struct bethparm {
  double zt;
  double zb;
  double at;
  double ab;
  double ipot;   /* Ionization potential in eV */
  double ene;  /* Energy in MeV */
  double th; /* Thickness in g/cm^2 */
  double eloss;
};

double fnbeth(struct bethparm *bp,double energy); /* Bethe-Block formula */
double eloss(struct bethparm *bp);
double ethick(struct bethparm *bp);
#endif

