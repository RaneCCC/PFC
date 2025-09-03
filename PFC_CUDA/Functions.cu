#include "Functions.cuh"

/* Determine dx_new, dy_new, dz_new under strain */
void Strain(float* eps, int time, InputControl InputC, InputPara InputP, SimPara& SimP)
{
	SimP.dx_new = (1. + eps[0] * time / InputC.totalTime) * InputP.dx;
	SimP.dy_new = (1. + eps[1] * time / InputC.totalTime) * InputP.dy;
	SimP.dz_new = (1. + eps[2] * time / InputC.totalTime) * InputP.dz;
}

/* For the calculation of correlation term in k space, need to call the PreCalcCorrelation function at the beginning to set parameters */
__global__ void CalcCorrelation(cufftComplex* k0arr, cufftComplex* k1arr, cufftComplex* k2arr,
	cufftComplex* omegak0, cufftComplex* omegak1, cufftComplex* omegak2,
	CalcCorrelationPara CorrP, SimPara SimP)
{
	int index;
	int x, y, z, x_lim, y_lim, z_lim;
	double kx, ky, kz, k2, rk;			//k-space coordinate
	int NxP = SimP.Nx / 2 + 1;		//Padded Nx, gotten from the cufftR2C
	double kC1, omval;
	double Pi32 = 32 * Pi * Pi;

	//To save divisions
	x_lim = (NxP - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;
	double kx_fac = 2 * Pi / (SimP.Nx * SimP.dx_new);
	double ky_fac = 2 * Pi / (SimP.Ny * SimP.dy_new);
	double kz_fac = 2 * Pi / (SimP.Nz * SimP.dz_new);


	// Access data by steps if Nx > blockDim.x * gridDim.x, same for Ny and Nz
	for (int i = 0; i < z_lim; i++)
		for (int j = 0; j < y_lim + 1; j++)
			for (int k = 0; k < x_lim; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < NxP && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * NxP + z * NxP * SimP.Ny;	// Coalsced memory access
					if (z < SimP.Nz / 2.)								// Set positive and negative frequency space
						kz = z * kz_fac;
					else
						kz = (z - SimP.Nz) * kz_fac;
					if (y < SimP.Ny / 2.)
						ky = y * ky_fac;
					else
						ky = (y - SimP.Ny) * ky_fac;
					kx = x * kx_fac;		// No need for x due to the symmetry of frequency of D2Z FFT, only half frequency space is output in cufft

					k2 = kx * kx + ky * ky + kz * kz;
					rk = (k2 >= 0.0) ? sqrt(k2) : 0.0;

					//XPFC kernel: omval=1-C2(k)
					kC1 = CorrP.DW1 * exp(CorrP.PreC1 * (rk - CorrP.qC1) * (rk - CorrP.qC1));
					omval = 1. - kC1;

					// HIGH k CUTTOFF: OTHERWISE XPFC FREE ENERGY VARIES WILDLY WITH dx, DOESNT CONVERGE
					if (k2 > Pi32) omval = (1. - k2 * 2. / Pi32) * (1. - k2 * 2. / Pi32);

					if (omval > CorrP.omcut) omval = CorrP.omcut;
					if (CorrP.icons == 1) omval = k2 * omval;
					if (CorrP.iwave == 0)
					{
						k0arr[index].x = -CorrP.dtMn * k2;
						k0arr[index].y = k0arr[index].x;
						omegak0[index].x = 1./ (1. + CorrP.dtMn * omval);
						omegak0[index].y = omegak0[index].x;
						k1arr[index].x = k0arr[index].x;
						k1arr[index].y = k0arr[index].x;
						omegak1[index].x = omegak0[index].x;
						omegak1[index].y = omegak0[index].x;
						k2arr[index].x = k0arr[index].x;
						k2arr[index].y = k0arr[index].x;
						omegak2[index].x = omegak0[index].x;
						omegak2[index].y = omegak0[index].x;
					}
					else
					{
						k0arr[index].x = k2 / CorrP.dtalph2;
						k0arr[index].y = k0arr[index].x;
						k1arr[index].x = k0arr[index].x;
						k1arr[index].y = k0arr[index].x;
						k2arr[index].x = k0arr[index].x;
						k2arr[index].y = k0arr[index].x;
						
						omegak0[index].x = 1. / (1. - CorrP.dt * omval / CorrP.dtalph2);
						omegak0[index].y = omegak0[index].x;
						omegak1[index].x = omegak0[index].x;
						omegak1[index].y = omegak0[index].x;
						omegak2[index].x = omegak0[index].x;
						omegak2[index].y = omegak0[index].x;
					}
				}
			}
}

/* Set the coupling term gamma13 to choose the equilibrium phase(gamma13 > 0, H phase, gamma13 < 0, T phase) */
void SetCouplingTerm(int time, InputPara InputP, SimPara& SimP)
{
	int period = (InputP.t1 + InputP.t2) * 2;
	int kk = time % period;
	if (kk < period / 2.)
		SimP.gamma13 = (kk - period / 4.) * (-2 * InputP.gamma13 / InputP.t2);
	else
		SimP.gamma13 = (kk - 3 * period / 4.) * (2 * InputP.gamma13 / InputP.t2);
	if (SimP.gamma13 > InputP.gamma13)
		SimP.gamma13 = InputP.gamma13;
	else if (SimP.gamma13 < -InputP.gamma13)
		SimP.gamma13 = -InputP.gamma13;
}

/* Calculate the non - linear term in real space, need to call SetCouplingTerm first everytime to update gamma13 */
__global__ void CalcNL(float * n0, float* n1, float* n2, float* n0fNL, float* n1fNL, float* n2fNL,
	InputPara InputP, SimPara SimP)
{
	int index, x, y, z, x_lim, y_lim, z_lim;
	double couple0, couple1, couple2, dcouple0, dcouple1, dcouple2;
	double factor1, factor2, w, u;	//To save float divisions

	x_lim = (SimP.Nx - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;

	factor1 = 1.0 / InputP.sigmac;
	factor2 = factor1 * 0.5;
	w = -0.5 * InputP.w;
	u = InputP.u / 3.;

	// Access data by steps if Nx > blockDim.x * gridDim.x, same for Ny and Nz
	for (int i = 0; i < z_lim; i++)
		for (int j = 0; j < y_lim; j++)
			for (int k = 0; k < x_lim; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < SimP.Nx && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * SimP.Nx + z * SimP.Nx * SimP.Ny;	// Coalsced memory access
					couple0 = (1 + tanhf((n0[index] - InputP.nc) * factor1)) * 0.5;
					dcouple0 = 1. / (coshf((n0[index] - InputP.nc) * factor1) * coshf((n0[index] - InputP.nc) * factor1)) * factor2;
					couple1 = (1 + tanhf((n1[index] - InputP.nc) * factor1)) * 0.5;
					dcouple1 = 1. / (coshf((n1[index] - InputP.nc) * factor1) * coshf((n1[index] - InputP.nc) * factor1)) * factor2;
					couple2 = (1 + tanhf((n2[index] - InputP.nc) * factor1)) * 0.5;
					dcouple2 = 1. / (coshf((n2[index] - InputP.nc) * factor1) * coshf((n2[index] - InputP.nc) * factor1)) * factor2;
					n0fNL[index] = n0[index] * n0[index] * (w + u * n0[index]) + dcouple0 * (SimP.gamma12 * couple1 + SimP.gamma13 * couple2);
					n1fNL[index] = n1[index] * n1[index] * (w + u * n1[index]) + dcouple1 * (SimP.gamma12 * couple0 + SimP.gamma23 * couple2);
					n2fNL[index] = n2[index] * n2[index] * (w + u * n2[index]) + dcouple2 * (SimP.gamma13 * couple0 + SimP.gamma12 * couple1);
				}
			}
}

/* Update kn in kspace for damping wave dynamics */
__global__ void UpdateKnWave(cufftComplex* kn0, cufftComplex* kn1, cufftComplex* kn2,
	cufftComplex* kg0, cufftComplex* kg1, cufftComplex* kg2,
	cufftComplex* kn0fNL, cufftComplex* kn1fNL, cufftComplex* kn2fNL,
	cufftComplex* k0arr, cufftComplex* k1arr, cufftComplex* k2arr, 
	cufftComplex* omegak0, cufftComplex* omegak1, cufftComplex* omegak2,
	InputControl InputC, InputPara InputP, SimPara SimP)
{
	int index, x, y, z, x_lim, y_lim, z_lim;
	int NxP = SimP.Nx / 2 + 1;

	x_lim = (NxP - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;

	// First part of algorithm
	double rmult = InputP.dt * (1. - InputP.dt * InputC.betaw);

	// Access data by steps if Nx > blockDim.x * gridDim.x, same for Ny and Nz
	for (int i = 0; i < z_lim; i++)
		for (int j = 0; j < y_lim; j++)
			for (int k = 0; k < x_lim; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < NxP && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * NxP + z * NxP * SimP.Ny;	// Coalsced memory access
					kn0[index].x = omegak0[index].x * (kn0[index].x + InputP.dt * k0arr[index].x * kn0fNL[index].x) + rmult * omegak0[index].x * kg0[index].x;
					kn0[index].y = omegak0[index].y * (kn0[index].y + InputP.dt * k0arr[index].y * kn0fNL[index].y) + rmult * omegak0[index].x * kg0[index].y;
					kn1[index].x = omegak1[index].x * (kn1[index].x + InputP.dt * k1arr[index].x * kn1fNL[index].x) + rmult * omegak1[index].x * kg1[index].x;
					kn1[index].y = omegak1[index].y * (kn1[index].y + InputP.dt * k1arr[index].y * kn1fNL[index].y) + rmult * omegak1[index].y * kg1[index].y;
					kn2[index].x = omegak2[index].x * (kn2[index].x + InputP.dt * k2arr[index].x * kn2fNL[index].x) + rmult * omegak2[index].x * kg2[index].x;
					kn2[index].y = omegak2[index].y * (kn2[index].y + InputP.dt * k2arr[index].y * kn2fNL[index].y) + rmult * omegak2[index].y * kg2[index].y;
				}
			}
}

/* Update kn in kspace for dissipasive dynamics */
__global__ void UpdateKn(cufftComplex* kn0, cufftComplex* kn1, cufftComplex* kn2,
	cufftComplex* kn0fNL, cufftComplex* kn1fNL, cufftComplex* kn2fNL,
	cufftComplex* k0arr, cufftComplex* k1arr, cufftComplex* k2arr,
	cufftComplex* omegak0, cufftComplex* omegak1, cufftComplex* omegak2,
	InputControl InputC, InputPara InputP, SimPara SimP)
{
	int index, x, y, z, x_lim, y_lim, z_lim;
	int NxP = SimP.Nx / 2 + 1;

	x_lim = (NxP - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;

	// Access data by steps if Nx > blockDim.x * gridDim.x, same for Ny and Nz
	for (int i = 0; i < z_lim; i++)
		for (int j = 0; j < y_lim; j++)
			for (int k = 0; k < x_lim; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < NxP && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * NxP + z * NxP * SimP.Ny;	// Coalsced memory access
					kn0[index].x = omegak0[index].x * (kn0[index].x + InputP.dt * k0arr[index].x * kn0fNL[index].x);
					kn0[index].y = omegak0[index].y * (kn0[index].y + InputP.dt * k0arr[index].y * kn0fNL[index].y);
					kn1[index].x = omegak1[index].x * (kn1[index].x + InputP.dt * k1arr[index].x * kn1fNL[index].x);
					kn1[index].y = omegak1[index].y * (kn1[index].y + InputP.dt * k1arr[index].y * kn1fNL[index].y);
					kn2[index].x = omegak2[index].x * (kn2[index].x + InputP.dt * k2arr[index].x * kn2fNL[index].x);
					kn2[index].y = omegak2[index].y * (kn2[index].y + InputP.dt * k2arr[index].y * kn2fNL[index].y);
				}
			}
}

/* First Update g in real space (g = n_old) */
__global__ void UpdateG1(float* n0, float* n1, float* n2, float* g0, float* g1, float* g2, SimPara SimP)
{
	int index, x, y, z, x_lim, y_lim, z_lim;

	x_lim = (SimP.Nx - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;

	for (int i = 0; i < z_lim; i++)
		for (int j = 0; j < y_lim; j++)
			for (int k = 0; k < x_lim; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < SimP.Nx && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * SimP.Nx + z * SimP.Nx * SimP.Ny;	// Coalsced memory access
					g0[index] = n0[index];
					g1[index] = n1[index];
					g2[index] = n2[index];
				}
			}
}

/* Second update g in real space [g = (n_new - n_old)/dt] */
__global__ void UpdateG2(float* n0, float* n1, float* n2,
	float* g0, float* g1, float* g2, InputPara InputP, SimPara SimP)
{
	int index, x, y, z, x_lim, y_lim, z_lim;

	x_lim = (SimP.Nx - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;

	double factor = 1. / InputP.dt;

	for (int i = 0; i < z_lim; i++)
		for (int j = 0; j < y_lim; j++)
			for (int k = 0; k < x_lim; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < SimP.Nx && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * SimP.Nx + z * SimP.Nx * SimP.Ny;	// Coalsced memory access
					g0[index] = (n0[index] - g0[index]) * factor;
					g1[index] = (n1[index] - g1[index]) * factor;
					g2[index] = (n2[index] - g2[index]) * factor;
				}
			}
}