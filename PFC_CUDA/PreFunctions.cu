#include "PreFunctions.cuh"

/* Set the parameters at the beginning of simulations */
void SetParams(SimPara& SimP, InputPara& InputP)
{
	//from the lattice Spacing and grid SimPacing, compute the number of grid points
	//BCC
	SimP.Nx = (int)(floor(InputP.spacing / InputP.dx * InputP.atomsx + .5));
	SimP.Ny = (int)(floor(InputP.spacing / InputP.dy * InputP.atomsy + .5));
	SimP.Nz = (int)(floor(InputP.spacing / InputP.dz * InputP.atomsz + .5));

	//set up fourier scaling factors
	SimP.facx = 2. * Pi / (SimP.Nx * InputP.dx);
	SimP.facy = 2. * Pi / (SimP.Ny * InputP.dy);
	SimP.facz = 2. * Pi / (SimP.Nz * InputP.dz);

	//thermal noise amplitude, dt scaling
	//SimP.epdt = InputP.ampc/sqrt(InputP.dt);	//nonconserved dynamics
	SimP.epdt = InputP.ampc * sqrt(InputP.dt);	//conserved dynamics

	SimP.gamma12 = InputP.gamma12;
	SimP.gamma13 = InputP.gamma13;
	SimP.gamma23 = InputP.gamma23;

	SimP.print_para();
}

/* Allocate managed memory for variables */
void AllocateMemory(Variable& Var, InputControl InputC, SimPara SimP)
{
	int MeshSize = SimP.Nx * SimP.Ny * SimP.Nz;
	for (int i = 0; i < 3; i++)
	{
		cudaMallocManaged(&Var.n[i], sizeof(float) * SimP.Nx * SimP.Ny * SimP.Nz);// 统一内存
		cudaMallocManaged(&Var.nfNL[i], sizeof(float) * SimP.Nx * SimP.Ny * SimP.Nz);
		if (InputC.iwave == 1)
			cudaMallocManaged(&Var.g[i], sizeof(float) * SimP.Nx * SimP.Ny * SimP.Nz);

		cudaMallocManaged(&Var.kn[i], sizeof(cufftComplex) * ((SimP.Nx/2+1) * SimP.Ny * SimP.Nz));
		cudaMallocManaged(&Var.knfNL[i], sizeof(cufftComplex) * ((SimP.Nx / 2 + 1) * SimP.Ny * SimP.Nz));
		cudaMallocManaged(&Var.omegak[i], sizeof(cufftComplex) * ((SimP.Nx / 2 + 1) * SimP.Ny * SimP.Nz));
		cudaMallocManaged(&Var.karr[i], sizeof(cufftComplex) * ((SimP.Nx / 2 + 1) * SimP.Ny * SimP.Nz));
		cudaMallocManaged(&Var.kg[i], sizeof(cufftComplex) * ((SimP.Nx / 2 + 1) * SimP.Ny * SimP.Nz));
	}
}

/* Free managed memory for variables */
void FreeMemory(Variable& Var, InputControl InputC)
{
	for (int i = 0; i < 3; i++)
	{
 		cudaFree(Var.n[i]);
		cudaFree(Var.nfNL[i]);
		if (InputC.iwave)
			cudaFree(Var.g[i]);
		cudaFree(Var.kn[i]);
		cudaFree(Var.knfNL[i]);
		cudaFree(Var.omegak[i]);
		cudaFree(Var.karr[i]);
		cudaFree(Var.kg[i]);
	}
}

/* Set all cufftplan at the beginning of simulations */
void SetCufftplan(FFTHandle& FFTH, SimPara SimP)
{
	for (int i = 0; i < 3; i++)
	{
		CHECK(cufftPlan3d(&FFTH.planF_n[i], SimP.Nz, SimP.Ny, SimP.Nx, CUFFT_R2C));
		CHECK(cufftPlan3d(&FFTH.planB_n[i], SimP.Nz, SimP.Ny, SimP.Nx, CUFFT_C2R));
		CHECK(cufftPlan3d(&FFTH.planF_NL[i], SimP.Nz, SimP.Ny, SimP.Nx, CUFFT_R2C));
		CHECK(cufftPlan3d(&FFTH.planB_NL[i], SimP.Nz, SimP.Ny, SimP.Nx, CUFFT_C2R));
		CHECK(cufftPlan3d(&FFTH.planF_g[i], SimP.Nz, SimP.Ny, SimP.Nx, CUFFT_R2C));
	}
}

/* Destroy all cufftplan at the end of simulations */
void DestroyCufftPlan(FFTHandle& FFTH)
{
	for (int i = 0; i < 3; i++)
	{
		CHECK(cufftDestroy(FFTH.planF_n[i]));
		CHECK(cufftDestroy(FFTH.planB_n[i]));
		CHECK(cufftDestroy(FFTH.planF_NL[i]));
		CHECK(cufftDestroy(FFTH.planB_NL[i]));
		CHECK(cufftDestroy(FFTH.planF_g[i]));
	}
}

/* Preparation for the calculation of correlation function */
void PreCalcCorrelations(InputControl InputC, InputPara InputP, CalcCorrelationPara& CorrP)
{
	CorrP.omcut = InputP.omcut;
	CorrP.icons = InputC.icons;
	CorrP.iwave = InputC.iwave;
	CorrP.dt = InputP.dt;
	CorrP.dtMn = InputP.dt * InputC.Mn;
	CorrP.dtalph2 = -1. / InputP.dt / InputC.alphaw / InputC.alphaw;

	// XPFC kernel for BCC - 1 peak
	CorrP.qC1 = 2. * Pi * sqrt(3);		//first mode
	CorrP.PreC1 = -0.5 / (InputP.alpha1 * InputP.alpha1);
	CorrP.DW1 = exp(-0.5 * InputP.sigmaT * InputP.sigmaT * CorrP.qC1 * CorrP.qC1 / (InputP.rho1 * InputP.beta1));

}

/* Initialize n[] and g[] if no restart file is read */
/* Have to set parameters separately, can not be an array since this is in GPU as a kernel. The declaration of array is in main memory. */
__global__ void initialize(float* n0, float* n1, float* n2, float* g0, float* g1, float* g2, int phase, InputControl InputC, InputPara InputP, SimPara SimP)
{
	float theta1 = 5 * Pi / 180;	//Orientation of phase 1
	float theta2 = -5 * Pi / 180;	//Orientation of phase 2
	phase = -1;	/* 1 for H and -1 for T */
	float qx = 2. * Pi * InputP.dx * sqrt(3.);
	float qy = 2. * Pi * InputP.dy * sqrt(3.);
	float qz = 2. * Pi * InputP.dz * sqrt(3.);
	int x, y, z, index;
	float x1, y1;

	// Access data by steps if Nx > blockDim.x * gridDim.x, same for Ny and Nz
	for (int i = 0; i < (SimP.Nz-1) / (blockDim.z * gridDim.z) + 1; i++)
		for (int j = 0; j < (SimP.Ny-1) / (blockDim.y * gridDim.y) + 1; j++)
			for (int k = 0; k < (SimP.Nx-1) / (blockDim.x * gridDim.x) + 1; k++)
			{
				// 3D grids
				z = threadIdx.z + blockIdx.z * blockDim.z + i * blockDim.z * gridDim.z;
				y = threadIdx.y + blockIdx.y * blockDim.y + j * blockDim.y * gridDim.y;
				x = threadIdx.x + blockIdx.x * blockDim.x + k * blockDim.x * gridDim.x;

				if (x < SimP.Nx && y < SimP.Ny && z < SimP.Nz)
				{
					index = x + y * SimP.Nx + z * SimP.Nx * SimP.Ny;	// Coalsced memory access

					// Input BCC crystal
					if (y > SimP.Ny / 2. + 10 && y < SimP.Ny - 10)	// Region for phase 1
					{
						x1 = x * cos(theta1) + y * sin(theta1);
						y1 = -x * sin(theta1) + y * cos(theta1);
						n0[index] = InputP.amp0 * (cos(qx * (x1 - 4 * Pi / 3. / qx)) + cos(-qx * (x1 - 4 * Pi / 3. / qx) * sin(-Pi / 6.) + qy * y1 * cos(-Pi / 6.)) + cos(-qx * (x1 - 4 * Pi / 3. / qx) * sin(Pi / 6.) + qy * y1 * cos(Pi / 6.))) + InputP.n0;
						n1[index] = InputP.amp0 * (cos(qx * x1) + cos(qx * (-x1 * sin(-Pi / 6.) + y1 * cos(-Pi / 6.))) + cos(qx * (-x1 * sin(Pi / 6.) + y1 * cos(Pi / 6.)))) + InputP.n0;
						n2[index] = InputP.amp0 * (cos(qx * (x1 - phase * 4 * Pi / 3. / qx)) + cos(-qx * (x1 - phase * 4 * Pi / 3. / qx) * sin(-Pi / 6.) + qy * y1 * cos(-Pi / 6.)) + cos(-qx * (x1 - phase * 4 * Pi / 3. / qx) * sin(Pi / 6.) + qy * y1 * cos(Pi / 6.))) + InputP.n0;
					}
					else if (y < SimP.Ny / 2. - 10 && y > 10)	// Region for phase 2
					{
						x1 = x * cos(theta2) + y * sin(theta2);
						y1 = -x * sin(theta2) + y * cos(theta2);
						n0[index] = InputP.amp0 * (cos(qx * (x1 - 4 * Pi / 3. / qx)) + cos(-qx * (x1 - 4 * Pi / 3. / qx) * sin(-Pi / 6.) + qy * y1 * cos(-Pi / 6.)) + cos(-qx * (x1 - 4 * Pi / 3. / qx) * sin(Pi / 6.) + qy * y1 * cos(Pi / 6.))) + InputP.n0;
						n1[index] = InputP.amp0 * (cos(qx * x1) + cos(qx * (-x1 * sin(-Pi / 6.) + y1 * cos(-Pi / 6.))) + cos(qx * (-x1 * sin(Pi / 6.) + y1 * cos(Pi / 6.)))) + InputP.n0;
						n2[index] = InputP.amp0 * (cos(qx * (x1 - phase * 4 * Pi / 3. / qx)) + cos(-qx * (x1 - phase * 4 * Pi / 3. / qx) * sin(-Pi / 6.) + qy * y1 * cos(-Pi / 6.)) + cos(-qx * (x1 - phase * 4 * Pi / 3. / qx) * sin(Pi / 6.) + qy * y1 * cos(Pi / 6.))) + InputP.n0;
					}
					else			// Interfacial region
					{
						n0[index] = InputP.n0;
						n1[index] = InputP.n0;
						n2[index] = InputP.n0;
					}

					//Initialize g if wave dynamic used
					if (InputC.iwave == 1)
					{
						g0[index] = 0.0;
						g1[index] = 0.0;
						g2[index] = 0.0;
					}

				}
			}
}

/* Output n[] and g[] */
void output(float ** n, float ** g, InputControl InputC, SimPara SimP, int time)
{
	std::cout << "--------------Writting Output Now--------------" << std::endl;
	std::string OutputFileName_n[3], OutputFileName_g[3];
	for (int i = 0; i < 3; i++)
	{
		OutputFileName_n[i] = std::to_string(i+1) + "_" + InputC.run + "_" + std::to_string(time) + ".dat";
		if(InputC.iwave)
			OutputFileName_g[i] = "g" + std::to_string(i+1) + "_" + InputC.run + "_" + std::to_string(time) + ".dat";
	}

	std::ofstream outf_n, outf_g;
	for (int i = 0; i < 3; i++)
	{
		outf_n.open(OutputFileName_n[i]);
		if (!outf_n.is_open())
		{
			std::cout << "!!!!!Can not open" << OutputFileName_n[i] << "!! Exit!!!!!!!!" << std::endl;
			exit(1);
		}
		if (InputC.iwave)
		{
			outf_g.open(OutputFileName_g[i]);
			if (!outf_g.is_open())
			{
				std::cout << "!!!!!Can not open" << OutputFileName_g[i] << "!! Exit!!!!!!!!" << std::endl;
				exit(1);
			}
		}
	
		/* Writting output */
		std::cout << "Writting output into " << OutputFileName_n[i] << std::endl;
		
		for (int z=0; z < SimP.Nz; z++)
			for (int y=0; y < SimP.Ny; y++)
				for (int x=0; x < SimP.Nx; x++)
				{
					int index = x + y * SimP.Nx + z * SimP.Nx * SimP.Ny;
					outf_n << n[i][index] << std::endl;
					if (InputC.iwave)
						outf_g << g[i][index] << std::endl;
				}
		outf_n.close();
		if (InputC.iwave)
			outf_g.close();
	}
	std::cout << "--------------Output Done--------------" << std::endl;
}

/* Output data in k-space, mainly used for debugging */
void output_complex(std::string OutputFilePreName, InputControl InputC, SimPara SimP, int time, cufftComplex* data)
{
	std::cout << "--------------Writting Output Now--------------" << std::endl;
	std::string OutputFileName[2];
	OutputFileName[0] = OutputFilePreName + "_" + InputC.run + "_" + std::to_string(time) + "_R.dat";
	OutputFileName[1] = OutputFilePreName + "_" + InputC.run + "_" + std::to_string(time) + "_I.dat";
	
	std::ofstream outf[2];
	for (int i = 0; i < 2; i++)
	{
		outf[i].open(OutputFileName[i]);
		if (!outf[i].is_open())
		{
			std::cout << "!!!!!Can not open" << OutputFileName[i] << "!! Exit!!!!!!!!" << std::endl;
			exit(1);
		}
	}

	/* Writting output */
	std::cout << "Writting output into " << OutputFileName[0] << " and " << OutputFileName[1] << std::endl;
	int NxP = SimP.Nx / 2 + 1;			//Padded Nx, gotten from the cufftR2C
	for (int z = 0; z < SimP.Nz; z++)
		for (int y = 0; y < SimP.Ny; y++)
			for (int x = 0; x < NxP; x++)
			{
				int index = x + y * NxP + z * NxP * SimP.Ny;
				outf[0] << data[index].x << std::endl;
				outf[1] << data[index].y << std::endl;
			}

	for (int i = 0; i < 2; i++)
		outf[i].close();
	std::cout << "--------------Output Done--------------" << std::endl;
}

/* Normalization before inverse-FFT to avoid lost of precision */
__global__ void pre_normalize(cufftComplex* data, SimPara SimP)
{
	int index;
	int x, y, z, x_lim, y_lim, z_lim;
	int NxP = SimP.Nx / 2 + 1;		//Padded Nx, gotten from the cufftR2C
	//To save divisions
	x_lim = (NxP - 1) / (blockDim.x * gridDim.x) + 1;
	y_lim = (SimP.Ny - 1) / (blockDim.y * gridDim.y) + 1;
	z_lim = (SimP.Nz - 1) / (blockDim.z * gridDim.z) + 1;

	int size = SimP.Nx * SimP.Ny * SimP.Nz;
	double factor = 1. / size;

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
					data[index].x *= factor;
					data[index].y *= factor;
				}
			}
}

/* Normalization required after each inverse-FFT */
__global__ void normalize(float * data, SimPara SimP)
{
	int size = SimP.Nx * SimP.Ny * SimP.Nz;
	float factor = 1. / size;

	// Access data by steps if Nx*Ny*Nz > blockDim.x * gridDim.x
	for (int i = 0; i < (size-1) / (blockDim.x * gridDim.x) + 1; i++)
	{
		// 3D grids
		int index = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;
		if(index < size)
			data[index] *= factor;
	}
}

/* Read existing n[] and g[] output files to restart */
void restart(float** n, float** g, InputControl InputC, SimPara SimP, int time)
{
	std::cout << "--------------Restart! Reading Existing Output Now--------------" << std::endl;
	std::string InputFileName_n[3], InputFileName_g[3];
	for (int i = 0; i < 3; i++)
	{
		InputFileName_n[i] = std::to_string(i + 1) + "_" + InputC.restartrun + "_" + std::to_string(time) + ".dat";
		if (InputC.iwave)
			InputFileName_g[i] = "g" + std::to_string(i + 1) + "_" + InputC.restartrun + "_" + std::to_string(time) + ".dat";
	}
	std::ifstream inf_n, inf_g;
	for (int i = 0; i < 3; i++)
	{
		inf_n.open(InputFileName_n[i]);
		if (!inf_n.is_open())
		{
			std::cout << "!!!!!Can not open" << InputFileName_n[i] << "!! Exit!!!!!!!!" << std::endl;
			exit(1);
		}
		if (InputC.iwave)
		{
			inf_g.open(InputFileName_g[i]);
			if (!inf_g.is_open())
			{
				std::cout << "!!!!!Can not open" << InputFileName_g[i] << "!! Exit!!!!!!!!" << std::endl;
				exit(1);
			}
		}

		/* Reading */
		std::cout << "Reading existing output from " << InputFileName_n[i] << std::endl;

		for (int z = 0; z < SimP.Nz; z++)
			for (int y = 0; y < SimP.Ny; y++)
				for (int x = 0; x < SimP.Nx; x++)
				{
					int index = x + y * SimP.Nx + z * SimP.Nx * SimP.Ny;
					inf_n >> n[i][index];
					if (InputC.iwave)
						inf_g >> g[i][index];
				}
		inf_n.close();
		if (InputC.iwave)
			inf_g.close();
	}
	std::cout << "--------------Restart Done--------------" << std::endl;
}