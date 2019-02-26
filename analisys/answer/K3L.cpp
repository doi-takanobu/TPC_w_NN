//*********************************
//2017/12/15 Doi Takanobu P4
//
//運動学の計算(c=1)
//
//引数に
//入射粒子の運動エネルギー、入射粒子、ターゲット、散乱粒子、残留粒子
//の順に与える
//(原子核の名前の入力は大文字で　ex:12C,60Zn,...)
//ただし、n,p,d,tは例外
//
//
//入力待ちの状態になったら、
//散乱角(Lab系)、残留核の励起エネルギー
//の順に入力
//(励起エネルギーは省略可能)
//
//結果は
//Th3L(deg) : 散乱角(Lab系)
//Ex4(MeV) : 励起エネルギー
//Th3C(deg) : 散乱角(C.M.系)
//K3L(MeV) : 散乱粒子の運動エネルギー(Lab系)
//P3L(MeV) : 散乱粒子の運動量(Lab系)
//Gfactor : 微分断面積の変換係数(Lab系-->C.M.系)
//が表示される
//
//入力待ちの状態で何も入力せずに"return"キーを押すと終了する
//
//角度と励起エネルギーの書かれたファイルを作り、
//リダイレクトで標準入力に入れられる。
//その標準出力をリダイレクトでファイルに保存すれば、
//gnuplot でそのままグラフが作れる。
//
//
//依存関係
//K3L.cpp      : 端末からの入出力
//kinema.cpp   : 運動学の計算
//database.cpp : 原子核のデータの管理
//dataset.cpp  : 原子核のデータを保存
//nuclear.cpp  : 原子核クラスの定義
//*********************************

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <map>
#include <string>
#include "kinema.hpp"
#include "database.hpp"
#include "nuclear.hpp"

void h_t_u();
void K3L(int argc,char *argv[]);
void showparticle();
void cross(int argc,char *argv[]);
void Ex4toEi(int argc,char *argv[]);

int main(int argc,char *argv[])
{
  int option=0;
  string para;
  map <string,int> opt;
  opt["--help"] = 0;//show help (how to use)
  opt["-h"] = 0;
  opt["--kinema"] = 1;//calculate kinematics
  opt["-k"] = 1;
  opt["--particle"] = 2;//show particle data
  opt["-p"] = 2;
  opt["--cross"] = 3;//show gfactor
  opt["-c"] = 3;
  opt["--ex4"] = 4;//show gfactor
  opt["-e"] = 4;

  for(int i=1;i<argc;i++){
    if(argv[i][0] == '-'){
      option = i;
      para = argv[i];
      break;
    }
  }
  if(option == 0){
    cout << "Illigal format." << endl;
    cout << "Please type \"./K3L -h\" or \"./K3L --help\"" << endl;
    exit(EXIT_FAILURE);
  }
  /*
  for(int i=option;i<argc-1;i++){
    argv[i]=argv[i+1];
    }*/
  argv++;
  argc = argc-1;
  /*
  if(opt[para]==0){
    h_t_u();
    return 0;
  }
  else if(opt[para]==1){
    K3L(argc,argv);
    return 0;
  }
  else if(opt[para]==2){
    showparticle();
    return 0;
  }
  else if(opt[para]==3){
    cross(argc,argv);
    return 0;
  }
  else if(opt[para]==4){
    Ex4toEi(argc,argv);
    return 0;
  }
  else{
    cout << "Illigal format." << endl;
    cout << "Please type \"./K3L -h\" or \"./K3L --help\"" << endl;
    return 0;
  }
  */
  switch(opt[para]){
  case 0:
    h_t_u();
  case 1:
    K3L(argc,argv);
    break;
  case 2:
    showparticle();
    break;
  case 3:
    cross(argc,argv);
    break;
  case 4:
    Ex4toEi(argc,argv);
    break;
  default:
    cout << "Illigal format." << endl;
    cout << "Please type \"./K3L -h\" or \"./K3L --help\"" << endl;
    break;
  }
  
  return 0;
}

void h_t_u()
{
  cout << "-h/--help : このプログラムの使い方" << endl;
  cout << "-k/--kinema : 散乱による運動学の計算";
  cout << "[beam energy][proj][targ][ejec][reciol]" << endl;
  cout << "-p/--particle : 粒子のmass excess を表示 (引数なし)" << endl;
  cout << "-c/--cross : 断面積の変換係数の表示(Lab -> C.M.)";
  cout << "[beam energy][proj][targ][ejec][reciol]" << endl;
  cout << "-e/--ex4 : 散乱粒子のエネルギーと残留核の励起エネルギーから";
  cout << "入射エネルギーを計算";
  cout << "[proj][targ][ejec][reciol]" << endl;
  cout << endl;
  return;
}

void K3L(int argc,char *argv[])
{
  kinema a;
  int first=0;
  a.setenv(argc,argv);
  a.showparticles();
  a.showparameter();
  
  while(1){
    if(a.calc()==0){
      if(first==0){
	cout << "#" << " Th3L(deg) |" << " Ex4(MeV) |" << " Ex3(MeV) |";
	cout << " Th3C(deg) |" << " Th4L(deg) |";
	cout << " K3L(MeV) |" << " P3L(MeV) |" << " K4L(MeV) |"<< endl;
	first = 1;
      }
      cout << " " << setw(10) << right << a.getthr3() << "  ";
      cout << setw(9) << a.getEx2() << "  ";
      cout << setw(9) << a.getEx1() << " ";
      cout << setw(10) << a.getthr3_C() << "  ";
      cout << setw(10) << a.getthr4() << "  ";
      cout << setw(9) << right << a.getE_L(2) << "  ";
      cout << setw(9) << a.getp() << "  ";
      cout << setw(9) << a.getE_L(3) << " ";
      cout << endl;
    }else{
      cout << "thr_max = " << a.getthr3_max() << endl;
      cout << "Particle is not detected." << endl;
    }
  }
  return;
}

void showparticle()
{
  database a;

  while(a.read_stdin()==0){
    cout << "delta = ";
    cout << a.get_delta(a.get_A(),a.get_name()) << "MeV/c^2" << endl;
    cout << "mass = ";
    cout << a.get_A()*amu+a.get_delta(a.get_A(),a.get_name());
    cout << "MeV/c^2" << endl;
  }
}

void cross(int argc,char *argv[])
{
  kinema a;
  int first=0;
  if(argc==6){
    a.setparticles(argv[2],argv[3],argv[4],argv[5]);
    a.setE_inc((double)strtol(argv[1],NULL,10));
  }else{
    cout << "Illigal format." << endl;
    h_t_u();
    exit(EXIT_FAILURE);
  }
  a.showparticles();
  a.showparameter();
  
  while(1){
    if(a.calc()==0){
      if(first==0){
	cout << "#" << " Th3L(deg) |" << " Ex4(MeV) |" << " Th3C(deg) |";
	cout << " Gfactor |" << endl;
	first = 1;
      }
      cout << " " << setw(10) << right << a.getthr3() << "  ";
      cout << setw(9) << a.getEx2() << "  ";
      cout << setw(10) << a.getthr3_C() << "  ";
       cout << setw(8) << a.get_gfactor() << "  ";
      cout << endl;
    }else{
      cout << "thr_max = " << a.getthr3_max() << endl;
      cout << "Particle is not detected." << endl;
    }
  }
  return;
}

void Ex4toEi(int argc,char *argv[])
{
  kinema a;
  int first=0;
  if(argc==5){
    a.setparticles(argv[1],argv[2],argv[3],argv[4]);
  }else{
    cout << "Illigal format." << endl;
    h_t_u();
    exit(EXIT_FAILURE);
  }
  a.showparticles();
  a.showparameter();
  
  while(1){
    if(a.ex4toei()==0){
      if(first==0){
	cout << "#" << " Th3L(deg) |" << " Ex4(MeV) |";
	cout << " K2(MeV) |" << " E_inc(MeV) |";
	cout << endl;
	first = 1;
      }
      cout << " " << setw(10) << right << a.getthr3() << "  ";
      cout << setw(9) << a.getEx2() << "  ";
      cout << setw(8) << a.getE_L(2) << "  ";
      cout << setw(11) << a.getE_L(0) << "  ";
      cout << endl;
    }else{
    }
  }
  return;
}
