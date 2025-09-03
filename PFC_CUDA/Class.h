/*
定义了多个类，用于存储系统控制输入参数、模拟输入参数、模拟过程中使用的其他参数、演化变量、FFT 以及相关计算参数。
这些类包括 InputControl、InputPara、SimPara、Variable、FFTHandle 和 CalcCorrelationPara，每个类都有相应成员变量和打印函数。
依赖：包含 <iostream> 和 cufft.h 头文件。
*/
#ifndef _CLASS_H_
#define _CLASS_H_

#include <iostream>
#include "cufft.h"

#define Pi (2.*acos(0.))

/* Definition of all the required classes */

/* System control input parameters */
class InputControl
{
	public:
		std::string run;	//label of current run
		int totalTime, printFreq;	//simulation time, print freq, ene freq
		int restartFlag, restartTime;		//flag for restarting (0=new run, 1=restart from old run) and time
		std::string restartrun;		//label of the run restarting from
		int icons, iwave;		//Conserved dynamics (0=nonconserved, 1=conserved)
		double Mn, alphaw, betaw;		//Wave dynamics (0=dissipative, 1=damped wave)

		void print_input() {
			printf("--------------System control input parameters--------------\n");
			printf("Label of current run: %s\n", &run[0]);
			printf("totalTime = %d,	printFreq = %d\n", totalTime, printFreq);
			printf("icons = %d, Mn = %f\n", icons, Mn);
			printf("iwave = %d,	alphaw = %lf, betaw = %lf\n", iwave, alphaw, betaw);
			printf("restartFlag = %d, restartTime = %d\n", restartFlag, restartTime);
			printf("restartrun = %s\n", &restartrun[0]);
		};
};

/* Input parameters for simulations */
class InputPara
{
	public:
		double atomsx, atomsy, atomsz;	//number of cells in x,y and z
		double spacing;			//lattice spacing
		double dx, dy, dz;		//numerical grid spacings
		double n0;			//avg density value
		int itemp;			//thermal noise switch
		double ampc;		//thermal noise amplitude unscaled
		double dt;				//time step
		double amp0, icpower;	//amp0=amplitude, icpower=exponent
		double w, u;		//-n^3/6 coeff (w), n^4/12 coefficient (u)
		double gamma12, gamma13, gamma23, nc, sigmac;         //Coupling coefficients between layers
		double sigmaT, omcut;	// DebyeWaller Temperature, omcut=min C2 value
		double alpha1;			// Widths of 1st C2 peak
		double rho1;			// Density of 1st family of planes
		double beta1; 			// Number of planes within 1st family of planes
		double eps[3];		//initial strain in x, y, and z directions
		double eps_v[3];	//varing strain in x, y, and z directions
		int gamma13_switch, t1, t2;		//if switch the sign of gamma13 periodically, switch period t1 and t2

		void print_input() {
			printf("--------------Simulation input parameters--------------\n");
			printf("numAtomsx = %lf, numAtomsy = %lf, numAtomsz = %lf,\n", atomsx, atomsy, atomsz);
			printf("spacing = %lf, dx = %lf, dy = %lf, dz = %lf, dt = %lf\n", spacing, dx, dy, dz, dt);
			printf("n0 = %lf\n", n0);
			printf("itemp = %d, ampc = %lf, amp0 = %lf, icpower = %lf\n", itemp, ampc, amp0, icpower);
			printf("w = %lf, u = %lf, gamma12 = %lf, gamma13 = %lf, gamma23 = %lf, nc = %lf, sigmac = %lf\n", w, u, gamma12, gamma13, gamma23, nc, sigmac);
			printf("sigmaT = %lf, omcut = %lf\n", sigmaT, omcut);
			printf("alpha1 = %lf, rho1 = %lf, beta1 = %lf\n", alpha1, rho1, beta1);
			printf("eps_x = %lf, eps_y = %lf, eps_z = %lf\n", eps[0], eps[1], eps[2]);
			printf("eps_x_varing = %lf, eps_y_varing = %lf, eps_z_varing = %lf\n", eps_v[0], eps_v[1], eps_v[2]);
			printf("gamma13_switch = %d, t1 = %d, t2 = %d\n", gamma13_switch, t1, t2);
		};
};

/* Other parameters used in simulation */
class SimPara
{
	public:
		int Nx, Ny, Nz;		//number of grid points in x,y and z
		double dx_new, dy_new, dz_new;          //numerical grid spacings under strain
		double facx, facy, facz;	//fourier factors for scaling lengths in k-space
		double epdt;	//thermal noise amplitude scaled
		double gamma12, gamma13, gamma23;
		void print_para() {
			printf("--------------Simulation parameters--------------\n");
			printf("Nx = %d, Ny = %d, Nz = %d\n", Nx, Ny, Nz);
			printf("facx = %lf, facy = %lf, facz = %lf\n", facx, facy, facz);
			printf("epdt = %lf\n", epdt);
			printf("gamma12 = %lf, gamma13 = %lf, gamma23 = %lf\n", gamma12, gamma13, gamma23);
		}
		void print_grid_under_strain() {
			printf("--------------Grid under strain--------------\n");
			printf("dx_new = %lf, dy_new = %lf, dz_new = %lf\n", dx_new, dy_new, dz_new);
		}
};

/* Variables evolved during simulations */
class Variable
{
	public:
		float* n[3], * nfNL[3], * g[3];
		cufftComplex* kn[3], * knfNL[3], * omegak[3], * karr[3], *kg[3];
};

/* All Handles of cufft */
class FFTHandle
{
	public:
		cufftHandle planF_n[3], planB_n[3], planF_NL[3], planB_NL[3], planF_g[3];
};

/* Parameters calculated by correlation function precalculations */
class CalcCorrelationPara
{
	public:
		double dtMn, dtalph2, qC1, PreC1, DW1, omcut, icons, iwave, dt;
};

#endif
