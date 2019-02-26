/*
98/10/19 revised by Kawabata Takahiro
*/

#include <math.h>
#include <stdio.h>
#include "kinema.h"
#include "bethe.h"

double fnbeth(struct bethparm *bp,double energy){ /* Bethe-Block formula */
  double factor;
  double massb,beta,gamma;
  
  double wmax,eta;
    
  double logpart;
  
  double dedx;
  
  massb=AMU*bp->ab;
  gamma=(energy+massb)/massb;
  beta=sqrt(1-1/(gamma*gamma));
  
  factor=FAC*bp->zt/bp->at*pow(bp->zb/beta,2.);
  
  eta=beta*gamma;
  wmax=2*pow(eta,2.)*MASSE;
  
  logpart=log(pow(wmax*MEGA/bp->ipot,2.));
  
  dedx=factor*(logpart-2*pow(beta,2.));
  return(dedx);
}

//////////////////////////////////
// double eloss(struct bethparm *bp) 
// Function to calculate the target thickness from energy loss
// Input
// bp->zt ... Target atomic number
// bp->za ... Target mass number
// bp->bt ... Beam atomic number
// bp->ba ... Beam mass number
// bp->ene ... Beam energy in MeV
// bp->th ... Target thickness in g/mg^2
// bp->ipot ... Ionizing potential in eV
//
// Output
// bp->eloss ... Beam energy loss in the target in MeV
//
// Return value
// Beam energy loss in the target in MeV
double eloss(struct bethparm *bpara) {
  double energy,delta_ei;   /*データ入力のための変数*/

  int i,n;                    /*数値積分のための変数*/
  double delta_e,step_e,min_delta,max_delta,delta_err;
  
  double thick,midE,dx,dcheck;  /*中点法のための変数*/
  
  double ene_l,ene_h,dx_l,dx_h,kawa,bata;  /*台形法のための変数*/
  
  double taka,hiro;  /*Simpson 法のための変数*/
  
  //  printf("Energy [MeV] ?:");
  //  scanf("%lf",&energy);
  //  printf("Z of target ?:");
  //  scanf("%lf",&ztarget);
  //  printf("A of target ?:");
  //  scanf("%lf",&atarget);
  //  printf("z of beam ?:");
  //  scanf("%lf",&zbeam);
  //  printf("a of beam ?:");
  //  scanf("%lf",&abeam);
  //  printf("電離 Potential [eV]?:");
  //  scanf("%lf",&ipot);
  //  printf("Thickness [g/cm^2]?: ");
  //  scanf("%lf",&thickness);
  //  printf("Energy Step [eV]?:");
  //  scanf("%lf",&delta_ei);
  delta_ei=BETH_ESTEP;
  //  printf("分割数?:");
  //  scanf("%d",&n);
  n=BETH_BIN;
  delta_ei/=MEGA;
  delta_e=10;  /* Expected energy loss in the target. */
 /* Initial value = 10 MeV */
 /* Calculation is processed until the difference between
    delta_e and actual energy loss in the target is smaller 
    than delta_ei. */

  energy=bpara->ene;
  min_delta=0;
  max_delta=energy;
  delta_err=max_delta-min_delta;
  
  while(delta_ei < delta_err){
  dcheck=0;
  ene_h=energy;
  bata=0;
  hiro=0;
  step_e=delta_e/(2*n);
  
  for(i=0;i<n;i++){
    midE=energy-(2*i+1)*step_e;           /*中点法*/
    dx=1/fnbeth(bpara,midE);
    thick=dx*2*step_e;
    dcheck=dcheck+thick;
    
    ene_l=ene_h-2*step_e;                     /*台形法*/
    dx_l=1/fnbeth(bpara,ene_l);
    dx_h=1/fnbeth(bpara,ene_h);
    kawa=(dx_l+dx_h)*step_e;
    bata=kawa+bata;
    ene_h=ene_l;
    
    taka=(thick*2+kawa)/3;         /*Simpson法*/
    hiro=taka+hiro;
  }
  if(hiro<bpara->th) min_delta=delta_e;
  else max_delta=delta_e;
  
  printf("min_delta:%f   max_delta:%f  deltae:%f\n",min_delta,max_delta,delta_e);
  delta_err=max_delta-min_delta;
  delta_e=min_delta+delta_err/2.0;
  }

  //  printf("Simpson= %e [MeV]\n",delta_e);
  bpara->eloss=delta_e;
  return(delta_e);
}

#if 0
int main(){
  struct bethparm bparm;
  bparm.zt=6;
  bparm.at=12;
  bparm.zb=2;
  bparm.ab=4;
  bparm.ipot=72;
  bparm.ene=10;
  bparm.eloss=1;
  printf("thick:%f\n",ethick(&bparm));
}
#endif

//////////////////////////////////
// double ethick(struct bethparm *bp) 
// Function to calculate the target thickness from energy loss
// Input
// bp->zt ... Target atomic number
// bp->za ... Target mass number
// bp->bt ... Beam atomic number
// bp->ba ... Beam mass number
// bp->ene ... Beam energy in MeV
// bp->eloss ... Beam energy loss in the target in MeV
// bp->ipot ... Ionizing potential in eV
//
// Output
// bp->th ... Target thickness in g/mg^2
//
// Return value
// Traget thickness in g/cm^2
double ethick(struct bethparm *bpara) {
  double energy,delta_e;   /*データ入力のための変数*/
  
  int n,i;                    /*数値積分のための変数*/
  double step_e;
  
  double thick,midE,dx,dcheck;  /*中点法のための変数*/
  
  double ene_l,ene_h,dx_l,dx_h,kawa,bata;  /*台形法のための変数*/
  
  double taka,hiro;  /*Simpson 法のための変数*/
  
  /*
  printf("Energy [MeV] ?:");
  scanf("%lf",&energy);
  printf("Z of target ?:");
  scanf("%lf",&ztarget);
  printf("A of target ?:");
  scanf("%lf",&atarget);
  printf("z of beam ?:");
  scanf("%lf",&zbeam);
  printf("a of beam ?:");
  scanf("%lf",&abeam);
  printf("電離 Potential [eV]?:");
  scanf("%lf",&ipot);
  printf("Energy Loss [MeV]?: ");
  scanf("%lf",&delta_e);*/
  //  printf("分割数 ?:");
  //  scanf("%d",&n);

  n=BETH_BIN;
  energy=bpara->ene;
  delta_e=bpara->eloss;
  dcheck=0;
  step_e=delta_e/(2*n);
  
  ene_h=energy;

  bata=0;
  hiro=0;

  for(i=0 ; i<n ;++i) {
    midE=energy-(2*i+1)*step_e;/*中点法*/
    if(midE<0.0001){
      break;
    }
    dx=1/fnbeth(bpara,midE);
    thick=dx*2*step_e;
    dcheck=dcheck+thick;
    
    ene_l=ene_h-2*step_e;                     /*台形法*/
    dx_l=1/fnbeth(bpara,ene_l);
    dx_h=1/fnbeth(bpara,ene_h);
    kawa=(dx_l+dx_h)*step_e;
    bata=kawa+bata;
    ene_h=ene_l;
    
    taka=(thick*2+kawa)/3;         /*Simpson法*/
    hiro=taka+hiro;
  }

  /*printf("中点法=  %e [g/cm^2]\n",dcheck);
  printf("台形法=  %e [g/cm^2]\n",bata);
  printf("Simpson= %e [g/cm^2]\n",hiro);*/

  bpara->th=dcheck;
  return(dcheck);
}
