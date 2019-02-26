{
  //  char* filename = "indirect_norm-7.root";
  TFile *file = new TFile("answer.root");
  //  char* cmd = "";
  //  sprintf(cmd,"Ex>>Ex(100,-3,3)",histname);
  pred_out->Draw("Ex>>Ex(100,-3,3)");
  Ex->SetTitle(";Ex (MeV);");
  Ex->SetTitleSize(0.05,"x");
  Ex->SetTitleSize(0.05,"y");
  Ex->SetLabelSize(0.05,"x");
  Ex->SetLabelSize(0.05,"y");
}
