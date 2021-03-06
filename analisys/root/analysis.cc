#include "analysis.hh"
#include "bethe.h"
#include "bethe_doi.h"
#include "kinema.h"
#include "kinema.hpp"
#include <iostream>
#include <cmath>
#include <TROOT.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSpline.h>
#include <fstream>

Double_t getE(Double_t xv1,Double_t xv3,Double_t xs1,Double_t xs3,TSpline3* spline)
{
  struct bethparm bp = {2,2,4,4,54.4,0,0,0};
  Double_t E=1.;
  Double_t dE=1.;
  Double_t dx;

  dx = sqrt((xs1-xv1)*(xs1-xv1)+(xs3-xv3)*(xs3-xv3));
  dx = dx*ch_to_cm;
  dx = dx*rho_He;// g/cm2
  
  return getE(dx,spline);
}

Double_t getE(Double_t dx,TSpline3* spline)
{
//  struct bethparm bp = {2,2,4,4,54.4,0,0,0};
//  const double de = 0.001;
//  double E = 0.1;
//  double range = 0;
//  double rho = 0.00011647;//density He (96%) + CO2 (4%) @ 0.5atm calc by LISE++
//  dx = dx*ch_to_cm;
//  bp.ene = E;
//  while(1){
//    range += 1./(bethe(2,4,1,2,4,E)*384./560.+
//		 bethe(6,12,1,2,4,E)*48./560.+
//		 bethe(8,16,1,2,4,E)*128./560.)*de/rho;
//    std::cout << "range: " << range << std::endl;
//    std::cout << "E: " << E << std::endl;
//    if(range>dx){
//      E -= de;
//      break;
//    }
//    E += de;
//  }

//  //fitting curve with calced data by LISE++
//  double a = -0.004136987572489;
//  double b = 0.054341502222524;
//  double c = 0.033983538822044;
//  E = a*pow(dx,3)+b*pow(dx,2)+c*pow(dx,1);

//  dx = dx*ch_to_cm;
//  double a = 0.437067159459644;
//  double b = 1.70613516206486;
//  double c = 0.889015901255412-dx;
//
//  double E = (-b+sqrt(b*b-4*a*c))/(2*a);
  
  return spline->Eval(dx);
}

Double_t getthr(Double_t xv1,Double_t xv3,Double_t xs1,Double_t xs3)
{
  Double_t thr;
  Double_t dx;

  dx = sqrt((xs1-xv1)*(xs1-xv1)+(xs3-xv3)*(xs3-xv3));
  thr = acos((xs1-xv1)/dx);
  
  return thr;
}

Double_t getEx(Double_t E,Double_t thr,Double_t m1,Double_t m2,Double_t m3,Double_t m4)
{
  Double_t Ex;
  const Double_t K1 = 750;

  Ex = calcex4(thr,m1,m2,m3,m4,K1,E);
  
  return Ex;
}

TTree* calc_Ex(TTree *tin,const char *treenam)
{
  TTree *tree = new TTree(treenam,treenam);
  
  Double_t Ex;
  Double_t E;
  Double_t dx;
  Double_t thr;
  Double_t xv1,xv3,xs1,xs3;
  
//  tin->SetBranchAddress("xv1",&xv1);
//  tin->SetBranchAddress("xv3",&xv3);
//  tin->SetBranchAddress("xs1",&xs1);
//  tin->SetBranchAddress("xs3",&xs3);
  tin->SetBranchAddress("dx",&dx);
  tin->SetBranchAddress("thr",&thr);
  tree->Branch("Ex",&Ex,"Ex/D");
  tree->Branch("E",&E,"E/D");
  tree->Branch("thr",&thr,"thr/D");

  kinema a;
  char *c = "10c";
  char *he = "4he";
  a.setparticles(c,he,he,c);
  const Int_t N = tin->GetEntries();

  TSpline3* spline = getSpline();

  for(Int_t ientry=0;ientry<N;ientry++){
    tin->GetEntry(ientry);
//    E = getE(xv1,xv3,xs1,xs3);
//    thr = getthr(xv1,xv3,xs1,xs3);
    thr = thr*M_PI/180.;
    E = getE(dx,spline);
    Ex = getEx(E,thr,a.getmass(0),a.getmass(1),a.getmass(2),a.getmass(3));
    tree->Fill();
    if(ientry%100==0){
      std::cout << "ientry: " << ientry << std::endl;
    }
  }

  return tree;
}

TSpline3* getSpline()
{
  
  // ******* spline calced by SRIM ********* //
  std::ifstream ifs("range_to_ene_500.dat");
  double range,MeV;
  char str[1024];
  TGraph* graph = new TGraph();
  if(ifs.fail()){
    std::cerr << "Failed to open file of range_to_ene_500.dat" << std::endl;
    exit(EXIT_FAILURE);
  }
  int i = 0;
  while(ifs.getline(str,1024-1)){
    sscanf(str,"%lf %lf",&range,&MeV);
    graph->SetPoint(i++,range,MeV);
  }
  ifs.close();
  
  TSpline3* spline = new TSpline3("newSpline3",graph);

  return spline;
}
