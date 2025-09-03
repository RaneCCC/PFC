/*
负责读取和打印输入参数。Input.h 中声明了 ReadInput 和 PrintInput 函数，Input.cpp 实现了这两个函数。
ReadInput 从指定的输入文件中读取系统控制和模拟所需的参数，将其存储在 InputControl 和 InputPara 类的对象中；
PrintInput打印输入参数。
依赖：依赖于 Class.h 中定义的 InputControl 和 InputPara 类。
*/

#include "Input.h"

void ReadInput(std::string InputFileName, InputControl& InputC, InputPara& InputP)
{
	std::ifstream infile;

	/* Open input file */
	infile.open(InputFileName);
	if (!infile.is_open())
	{
		std::cout << "!!!!!Can not open" << InputFileName << "!! Exit!!!!!!!!" << std::endl;
		exit(1);
	}

	std::string space;
	/* Read all input */
	std::cout << "Reading input from " << InputFileName << std::endl;
	infile >> InputC.run;
	std::getline(infile, space);
	infile >> InputP.atomsx >> InputP.atomsy >> InputP.atomsz;
	std::getline(infile, space);
	infile >> InputP.spacing >> InputP.dx >> InputP.dy >>InputP.dz >> InputP.dt;
	std::getline(infile, space);
	infile >> InputP.amp0 >> InputP.icpower;
	std::getline(infile, space);
	infile >> InputP.n0;
	std::getline(infile, space);
	infile >> InputC.totalTime >> InputC.printFreq;
	std::getline(infile, space);
	infile >> InputC.icons >> InputC.Mn;
	std::getline(infile, space);
	infile >> InputC.iwave >> InputC.alphaw >> InputC.betaw;
	std::getline(infile, space);
	infile >> InputP.itemp >> InputP.ampc;
	std::getline(infile, space);
	infile >> InputP.alpha1 >> InputP.rho1 >> InputP.beta1;
	std::getline(infile, space);
	infile >> InputP.sigmaT >> InputP.omcut;
	std::getline(infile, space);
	infile >> InputP.w >> InputP.u;
	std::getline(infile, space);
	infile >> InputP.gamma12 >> InputP.gamma13 >> InputP.gamma23 >> InputP.nc >> InputP.sigmac;
	std::getline(infile, space);
	infile >> InputC.restartFlag >> InputC.restartTime;
	std::getline(infile, space);
	infile >> InputC.restartrun;
	std::getline(infile, space);
	infile >> InputP.eps[0] >> InputP.eps[1] >> InputP.eps[2];
	std::getline(infile, space);
	infile >> InputP.eps_v[0] >> InputP.eps_v[1] >> InputP.eps_v[2];
	std::getline(infile, space);
	infile >> InputP.gamma13_switch >> InputP.t1 >> InputP.t2;
	std::getline(infile, space);
	
	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

}

void PrintInput(InputControl InputC, InputPara InputP)
{
	std::cout << "Input parameters are:" << std::endl;
	InputC.print_input();
	InputP.print_input();
}