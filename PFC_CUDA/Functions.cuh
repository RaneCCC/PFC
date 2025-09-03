#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <iostream>
#include "cuda_runtime.h"
#include "Class.h"

/* Determine dx_new, dy_new, dz_new under strain */
void Strain(float * eps, int time, InputControl InputC, InputPara InputP, SimPara& SimP);

/* For the calculation of correlation term in k space, need to call the PreCalcCorrelation function at the beginning to set parameters */
__global__ void CalcCorrelation(cufftComplex* k0arr, cufftComplex* k1arr, cufftComplex* k2arr,
	cufftComplex* omegak0, cufftComplex* omegak1, cufftComplex* omegak2,
	CalcCorrelationPara CorrP, SimPara SimP);

/* Set the coupling term gamma13 to choose the equilibrium phase(gamma13 > 0, H phase, gamma13 < 0, T phase) */
void SetCouplingTerm(int time, InputPara InputP, SimPara& SimP);

/* Calculate the non - linear term in real space, need to call SetCouplingTerm first everytime to update gamma13 */
__global__ void CalcNL(float* n0, float* n1, float* n2, float* n0fNL, float* n1fNL, float* n2fNL,
	InputPara InputP, SimPara SimP);

/* Update kn in kspace for damping wave dynamics */
__global__ void UpdateKnWave(cufftComplex* kn0, cufftComplex* kn1, cufftComplex* kn2,
	cufftComplex* kg0, cufftComplex* kg1, cufftComplex* kg2,
	cufftComplex* kn0fNL, cufftComplex* kn1fNL, cufftComplex* kn2fNL,
	cufftComplex* k0arr, cufftComplex* k1arr, cufftComplex* k2arr,
	cufftComplex* omegak0, cufftComplex* omegak1, cufftComplex* omegak2,
	InputControl InputC, InputPara InputP, SimPara SimP);

/* Update kn in kspace for dissipasive dynamics */
__global__ void UpdateKn(cufftComplex* kn0, cufftComplex* kn1, cufftComplex* kn2,
	cufftComplex* kn0fNL, cufftComplex* kn1fNL, cufftComplex* kn2fNL,
	cufftComplex* k0arr, cufftComplex* k1arr, cufftComplex* k2arr,
	cufftComplex* omegak0, cufftComplex* omegak1, cufftComplex* omegak2,
	InputControl InputC, InputPara InputP, SimPara SimP);

/* First Update g in real space (g = n_old) */
__global__ void UpdateG1(float* n0, float* n1, float* n2, float* g0, float* g1, float* g2, SimPara SimP);

/* Second update g in real space [g = (n_new - n_old)/dt] */
__global__ void UpdateG2(float* n0, float* n1, float* n2, float* g0, float* g1, float* g2, InputPara InputP, SimPara SimP);

#endif