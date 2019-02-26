#include "analysis.hh"
#include <TROOT.h>

#include <TTree.h>
#include <TFile.h>
#include <iostream>
#include <cstdlib>

int main(int argc,char *argv[])
{
  if(argc!=3){
    std::cerr << "./MAIKo dat output-rootfile" << std::endl;
    exit(EXIT_FAILURE);
  }
  
//  TFile *fin = new TFile(argv[1],"READ");
//  std::cout << "Read root-file: " << "prediction.root" << std::endl;
  TFile *fout = new TFile(argv[2],"RECREATE");
  std::cout << "Creat root-file: " << "ana.root" << std::endl;
//  TTree *tin1 = (TTree*)fin->Get("tree1");
//  TTree *tin2 = (TTree*)fin->Get("tree2");
  //  TTree *tin = new TTree("tin","tin");
  //  tin->ReadFile("../../cpp/generator/cell_MAIKo.dat","xv1/D:xv2/D:xv3/D:xs1/D:xs2/D:xs3/D:xb1/D:xb2/D:xb3/D",',');
  TTree *ans_out;
  TTree *pred_out;
  TTree *tin = new TTree("tin","tin");
//  tin->ReadFile(argv[1],"ans_dx/D:ans_thr/D:pred_dx/D:pred_thr");
  tin->ReadFile(argv[1],"pred_dx/D:pred_thr/D:pred_phi/D");
//  tin->ReadFile(argv[1],"pred_dx/D:pred_thr/D");
//  tin->ReadFile(argv[1],"pred_thr/D:pred_phi/D:pred_dx/D:pred_e3/D:pred_ex4/D");
//  TTree *ans_in = new TTree("ans_in","ans_in");
  TTree *pred_in = new TTree("pred_in","pred_in");
  Double_t ans_dx,ans_thr,pred_dx,pred_thr,pred_phi;
//  ans_in->Branch("dx",&ans_dx,"dx/D");
//  ans_in->Branch("thr",&ans_thr,"thr/D");
  pred_in->Branch("dx",&pred_dx,"dx/D");
  pred_in->Branch("thr",&pred_thr,"thr/D");
  pred_in->Branch("phi",&pred_phi,"phi/D");
//  tin->SetBranchAddress("ans_dx",&ans_dx);
//  tin->SetBranchAddress("ans_thr",&ans_thr);
  tin->SetBranchAddress("pred_dx",&pred_dx);
  tin->SetBranchAddress("pred_thr",&pred_thr);
  tin->SetBranchAddress("pred_phi",&pred_phi);
  Int_t nEntries = tin->GetEntries();
  for(Int_t i=0;i<nEntries;i++){
    tin->GetEntry(i);
//    ans_in->Fill();
    pred_in->Fill();
  }

  std::cout << "Start calc_Ex" << std::endl;
//  ans_out = calc_Ex(ans_in,"ans_out");
  pred_out = calc_Ex(pred_in,"pred_out");
  std::cout << "End calc_Ex" << std::endl;

//  ans_in->Write();
  pred_in->Write();
//  ans_out->Write();
  pred_out->Write();
  std::cout << "Writing tree" << std::endl;
  //  tout = calc_Ex(tin,"tout");
  //  tout->Write();
  fout->Close();
  //  fin->Close();
  std::cout << "Close root-files" << std::endl;
  
  return 0;
}
