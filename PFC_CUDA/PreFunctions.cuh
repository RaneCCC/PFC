//Functions for the preparation of the simulation (out of the loop)

#ifndef PREFUNCTIONS_H_
#define PREFUNCTIONS_H_

#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "cufft.h"
#include <string>
#include "Class.h"
#include "CUDAControl.cuh"

/* Set the parameters at the beginning of simulations */
void SetParams(SimPara& SP, InputPara& IP);

/* Allocate managed memory for variables */
void AllocateMemory(Variable& Var, InputControl InputC, SimPara SimP);

/* Free managed memory for variables */
void FreeMemory(Variable& Var, InputControl InputC);

/* Set all cufftplan at the beginning of simulations */
void SetCufftplan(FFTHandle& FFTH, SimPara SimP);

/* Destroy all cufftplan at the end of simulations */
void DestroyCufftPlan(FFTHandle& FFTH);

/* Preparation for the calculation of correlation function */
void PreCalcCorrelations(InputControl InputC, InputPara InputP, CalcCorrelationPara& CorrP);

/* Initialize n[] and g[] if no restart file is read */
__global__ void initialize(float* n0, float* n1, float* n2, float* g0, float* g1, float* g2, int phase, InputControl InputC, InputPara InputP, SimPara SimP);

/* Output n[] and g[] */
void output(float ** n, float ** g, InputControl InputC, SimPara SimP, int time);

/* Output data in k-space, mainly used for debugging */
void output_complex(std::string OutputFilePreName, InputControl InputC, SimPara SimP, int time, cufftComplex* data);

/* Normalization required after each inverse-FFT */
__global__ void normalize(float* data, SimPara SimP);

/* Normalization before inverse-FFT to avoid lost of precision */
__global__ void pre_normalize(cufftComplex* data, SimPara SimP);

/* Read n[] and g[] output files to restart */
void restart(float** n, float** g, InputControl InputC, SimPara SimP, int time);

#endif