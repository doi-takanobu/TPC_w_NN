#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <stdio.h>
// Scattering event:0
// Background event:1

double accuracy(std::vector<int>,std::vector<int>);
double efficiency(std::vector<int>,std::vector<int>);
double purity(std::vector<int>,std::vector<int>);
double jaccard(std::vector<int>,std::vector<int>);

int main(int argc,char *argv[])
{
  std::vector<int> True;
  std::vector<int> Pred;
  int N = 256;
  char str[N];
  FILE *fp;

  if(argc != 3){
    return -1;
  }
  fp = fopen(argv[1],"r");
  if(fp == NULL){
    return -1;
  }
  while(fgets(str,N,fp) != NULL){
    True.push_back(atoi(str));
  }
  fclose(fp);

  fp = fopen(argv[2],"r");
  if(fp == NULL){
    return -1;
  }
  while(fgets(str,N,fp) != NULL){
    Pred.push_back(atoi(str));
  }
  fclose(fp);

  std::cout << "accuracy:" << accuracy(True,Pred) << std::endl;
  std::cout << "efficiency:" << efficiency(True,Pred) << std::endl;
  std::cout << "purity:" << purity(True,Pred) << std::endl;
  std::cout << "jaccard:" << jaccard(True,Pred) << std::endl;
    
  return 0;
}

double accuracy(std::vector<int> True,std::vector<int> Pred)
{
  double correct = 0;
  if(True.size() != Pred.size()){
    exit(EXIT_FAILURE);
  }
  int num = True.size();
  for(int i=0;i<num;i++){
    if(True[i] == Pred[i]){
      correct+=1;
    }
  }
  return correct/num;
}

double efficiency(std::vector<int> True,std::vector<int> Pred)
{
  double correct=0;
  double population=0;
  if(True.size() != Pred.size()){
    exit(EXIT_FAILURE);
  }
  int num = True.size();
  for(int i=0;i<num;i++){
    if(True[i]==0){
      population+=1;
      if(Pred[i]==0){
	correct+=1;
      }
    }
  }

  return correct/population;
}

double purity(std::vector<int> True,std::vector<int> Pred)
{
  double correct=0;
  double population=0;
  if(True.size() != Pred.size()){
    exit(EXIT_FAILURE);
  }
  int num = True.size();
  for(int i=0;i<num;i++){
    if(Pred[i]==0){
      population+=1;
      if(True[i]==0){
	correct+=1;
      }
    }
  }

  return correct/population;
}

double jaccard(std::vector<int> True,std::vector<int> Pred)
{
  double correct=0;
  double population=0;
  if(True.size() != Pred.size()){
    exit(EXIT_FAILURE);
  }
  int num = True.size();

  for(int i=0;i<num;i++){
    if(!(True[i]==1 && Pred[i]==1)){
      population+=1;
      if(True[i]==Pred[i]){
	correct+=1;
      }
    }
  }

  return correct/population;
}
