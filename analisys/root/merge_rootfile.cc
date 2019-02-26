#include "analysis.hh"
#include <TROOT.h>

#include <TTree.h>
#include <TFile.h>
#include <iostream>
#include <cstdlib>

int main(int argc,char *argv[])
{
  if(argc!=4){
    std::cout << "./merge_rootfile input1 input2 output" << std::endl;
    exit(EXIT_FAILURE);
  }

  TFile *fin1 = new TFile(argv[1],"READ");
  TFile *fin2 = new TFile(argv[2],"READ");
  TFile *fout = new TFile(argv[3],"RECREATE");

  TTree *tin1 = (TTree*)fin1->Get("tout");
  TTree *tin2 = (TTree*)fin2->Get("tout");
  TTree *tout1 = new TTree("tout1","tout1");
  TTree *tout2 = new TTree("tout2","tout2");

  Double_t dx1,thr1,dx2,thr2;

  tin1->SetBranchAddress("dx",&dx1);
  tin1->SetBranchAddress("thr",&thr1);
  tin2->SetBranchAddress("dx",&dx2);
  tin2->SetBranchAddress("thr",&thr2);
  tout1->Branch("dx",&dx1,"dx/D");
  tout1->Branch("thr",&thr1,"thr/D");
  tout2->Branch("dx",&dx2,"dx/D");
  tout2->Branch("thr",&thr2,"thr/D");

  std::cout << "Writing 1st root-file" << std::endl;
  Int_t nEntries = tin1->GetEntries();
  for(Int_t i=0;i<nEntries;i++){
    tin1->GetEntry(i);
    tout1->Fill();
  }
  std::cout << "Writing 2nd root-file" << std::endl;
  nEntries = tin2->GetEntries();
  for(Int_t i=0;i<nEntries;i++){
    tin2->GetEntry(i);
    tout2->Fill();
  }

  tout1->Write();
  tout2->Write();

  fout->Close();
  fin1->Close();
  fin2->Close();

  std::cout << "Finished root-file merging" << std::endl;
    
  return 0;
}
