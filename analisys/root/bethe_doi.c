#include "bethe_doi.h"

double bethe_1(int Z,int A,double rho_dx,int z,int a,double E)
{
  double E_inc;
  int i,power,dE;

  //initialize
  E_inc=E;
  power=keta(E);

  //近似的に求める
  for(i=3;i>-3;i--){
    for(dE=0;dE<10;dE++){
      if(E_inc+dE*pow(10,i)-bethe(Z,A,rho_dx,z,a,E_inc+dE*pow(10,i))>E){
	E_inc=E_inc+(dE-1)*pow(10,i);
	break;
      }
      if(dE==9){
	E_inc=E_inc+dE*pow(10,i);
      }
    }
  }
  


  return E_inc;
}

int keta(double number)
{
  if(number==0){
    return -10000;
  }else{
    return log10(number)+1;
  }
}


double bethe(int Z,int A,double rho_dx,int z,int a,double E)
{
  double dE,c,I,beta,m,gamma,W_max,Delta[200][100],u;

  //initialization
  /*
  Delta=14.9312;//3He
  */
  //質量の定義(Delta[A][Z])*******
  Delta[1][0]=8.07131689453125;//n
  Delta[1][1]=7.288970703125;//p
  Delta[2][1]=13.1357216796875;//d
  Delta[3][1]=14.9498095703125;//t
  Delta[3][2]=14.9312177734375;//3He
  Delta[4][2]=2.42491552734375;//4He
  Delta[6][3]=14.08687890625;//6Li
  Delta[7][3]=14.90710546875;//7Li
  Delta[12][6]=0;//12C
  Delta[13][6]=3.1250087890625;//13C
  Delta[14][6]=3.019892822265625;//14C
  Delta[14][7]=2.863416748046875;//14N
  Delta[15][7]=0.10143871307373047;//15N
  Delta[16][8]=-4.73700146484375;//16O
  Delta[17][8]=-0.8087634887695313;//17O
  Delta[18][8]=-0.7828156127929687;//18O
  Delta[19][9]=-1.4874442138671875;//19F
  //******************************
  u=931;//(MeV/c2)
  m=a*u+Delta[a][z];
  gamma=(E+m)/m;
  beta=sqrt(1-1/(gamma*gamma));
  if(Z<13){  
    I=Z*(12+7./Z)*pow(10,-6);
  }else{
    I=Z*(9.76+58.8*pow(Z,-1.19))*pow(10,-6);
  }
  c=0.1535;//(Mev cm2/g)全体の係数
  W_max=2*0.511*beta*beta*gamma*gamma;
  /*
  X0=0.2014;//Siの場合
  X1=2.87;
  m=3.25;
  a=0.1492;
  C0=-4.44;
  X=log10(beta*gamma);
  if(X<X0){
    delta=0;
  }else if(X<X1){
    delta=4.6052*X+C0+a*pow(X1-X,m);
  }
  else{
    delta=4.6052*X+C0;
  }
  etha=beta*gamma;
  C=(0.422377*pow(etha,-2)+0.0304043*pow(etha,-4)-0.00038106*pow(etha,-6))*pow(10,-6)*I*I+(3.85019*pow(etha,-2)-0.1667989*pow(etha,-4)+0.00157955*pow(etha,-6))*pow(10,-9)*I*I*I;
  */

  //Bethe-Bloch formular
  dE=c*rho_dx*Z*z*z/(A*beta*beta)*(log(2*0.511*gamma*gamma*beta*beta*W_max/(I*I))-2*beta*beta);

  return dE;
}

