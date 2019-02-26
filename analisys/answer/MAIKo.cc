#include "analysis.hh"
#include <TROOT.h>

#include <TTree.h>
#include <TFile.h>
#include <iostream>
#include <cstdlib>

int main(int argc,char *argv[])
{
  TFile *fout = new TFile(argv[argc-1],"RECREATE");
  std::cout << "Creat root-file: " << "ana.root" << std::endl;
  TTree *ans_out;
  TTree *ans_in = new TTree("ans_in","ans_in");
  Double_t theta,phi,range,e3,ex4;
  ans_in->Branch("range",&range,"range/D");
  ans_in->Branch("theta",&theta,"theta/D");
  ans_in->Branch("phi",&phi,"phi/D");
  ans_in->Branch("e3",&e3,"e3/D");
  ans_in->Branch("ex4",&ex4,"ex4/D");
  for(Int_t ii=1;ii<argc-1;ii++){
    TTree *tin = new TTree("tin","tin");
    tin->ReadFile(argv[ii],"theta/D:phi/D:range/D:e3/D:ex4/D:hoge[8]/D");
    tin->SetBranchAddress("range",&range);
    tin->SetBranchAddress("theta",&theta);
    tin->SetBranchAddress("phi",&phi);
    tin->SetBranchAddress("e3",&e3);
    tin->SetBranchAddress("ex4",&ex4);
    Int_t nEntries = tin->GetEntries();
    for(Int_t i=0;i<nEntries;i++){
      tin->GetEntry(i);
      ans_in->Fill();
    }
  }

  std::cout << "Start calc_Ex" << std::endl;
  ans_out = calc_Ex(ans_in,"ans_out");
  std::cout << "End calc_Ex" << std::endl;

  ans_in->Write();
  ans_out->Write();
  std::cout << "Writing tree" << std::endl;
  //  tout = calc_Ex(tin,"tout");
  //  tout->Write();
  fout->Close();
  //  fin->Close();
  std::cout << "Close root-files" << std::endl;
  
  return 0;
}
