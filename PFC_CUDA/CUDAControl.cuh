/*
 Some stuff for CUDA control, like error checking and CUDA launch parameters
*/

#ifndef CUDACONTROL_H_
#define CUDACONTROL_H_

#include <iostream>
#include "cuda_runtime.h"
#include "cufft.h"

// For cuda error check
#define CheckLastCudaError() checklastcudaerror(__FILE__, __LINE__)
inline void checklastcudaerror(std::string const file, int const line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Cuda error: " << cudaGetErrorString(err) << " at line " << line << " in file " << file << std::endl;
		exit(2);
	}
}

// For cufft check
#define CHECK(x) check((x), __FILE__, __LINE__)
inline void check(cufftResult err, std::string const file, int const line) {
	if (err != CUFFT_SUCCESS) {
		std::cerr << "Error: " << err << ", line: " << line << " in " << file << std::endl;
		exit(3);
	}
}

#endif // !CUDACONTROL_H_
