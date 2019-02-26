#ifndef _BETHE_DOI_H_
#define _BETHE_DOI_H_

#include <stdio.h>
#include <math.h>
/*************************************************************
Bethe-Bloch                          
    4 
Z: atomic number of absorbing material
A: atomic weight of absorbing material
rho_dx: density of absorbing material (g/cm2)
z: charge of incident particle in units of e
a: atomic wight of incident particle
E: energy of particle after energy loss (MeV)
*************************************************************/
//エネルギー損失前のエネルギーの逆算
double bethe_1(int Z,int A,double rho_dx,int z,int a,double E);
//エネルギー損失(dE)
double bethe(int Z,int A,double rho_dx,int z,int a,double E);
//数字の桁を計算
int keta(double number);



#endif

