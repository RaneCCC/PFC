#include "cuda_runtime.h"// 计时
#include <iostream>
#include "cufft.h"		// cuFFT库，gpu上执行快速傅立叶变换
#include "Input.h"		// 自定义头文件，读取 打印输入
#include "Class.h"		// 自定义头文件，定义变量类
#include "PreFunctions.cuh"// 预处理cuda核函数
#include "Functions.cuh"// cuda核函数
#include "CUDAControl.cuh"// cuda内存管理 错误检查 cuFFT计划设置

// 宏定义，用于简化多维数组的传递
#define n_pack Var.n[0], Var.n[1], Var.n[2]         // 实空间场变量n (密度场)
#define g_pack Var.g[0], Var.g[1], Var.g[2]         // 辅助场变量g (通常是时间导数或旧的n值)
#define omegak_pack Var.omegak[0], Var.omegak[1], Var.omegak[2] // k空间与频率相关的复数项 (PFC方程中的线性项)
#define nfNL_pack Var.nfNL[0], Var.nfNL[1], Var.nfNL[2]     // 非线性项nfNL在k空间的复数数组
#define kn_pack Var.kn[0], Var.kn[1], Var.kn[2]     // 场变量n在k空间的复数数组 (n的fftw)
#define kg_pack Var.kg[0], Var.kg[1]p, Var.kg[2]     // 辅助场变量g在k空间的复数数组 (g的fftw)
#define karr_pack Var.karr[0], Var.karr[1], Var.karr[2] // k空间波矢数组 (用于计算k模长等)
#define knfNL_pack Var.knfNL[0], Var.knfNL[1], Var.knfNL[2] // 非线性项nfNL的fftw


void main()
{
    /*---------------------性能测试---------------------------*/
    time_t t_start_tot, t_end_tot; 
    t_start_tot = clock();         // 记录开始时的cpu时间
    /*---------------------性能测试---------------------------*/

    // CUDA核函数启动参数定义
    dim3 dimGrid3D(16, 16, 1);    // 网格：16x16x1个线程块
    dim3 dimBlock3D(16, 16, 1);  // 线程块：16x16x1个线程（每个线程块256个线程）
    
	// 一维操作的核函数启动（归一化）
    dim3 dimGrid1D(256, 1, 1);   // 网格维度：256x1x1个线程块
    dim3 dimBlock1D(256, 1, 1); // 线程块维度：256x1x1个线程（每个线程块256个线程）

    // 定义参数变量
    InputControl InputC; // 输入控制参数（如是否重启、打印频率等）
    InputPara InputP;   // 输入物理参数（如模拟尺寸、时间步长、材料数值等）
    SimPara SimP;       // 模拟派生参数（从InputP计算而来，如实际空间步长、k空间步长等）

    // 读取输入 设置参数
    ReadInput("PFC_CUDA.in", InputC, InputP); // 从PFC_CUDA.in文件读取输入参数
    PrintInput(InputC, InputP);               // 打印读取到的输入参数
    SetParams(SimP, InputP);                  // 根据InputP设置SimP中的派生参数

    // 分配变量内存（managed memory 统一内存，自动迁移数据，统一指针访问）
    Variable Var; // Variable类实例，其成员是指向GPU上管理内存的指针
    AllocateMemory(Var, InputC, SimP); // 分配GPU（管理）内存
    CheckLastCudaError();

    // 定义cuFFT句柄并设置cuFFT
    FFTHandle FFTH; 
    SetCufftplan(FFTH, SimP); // 根据模拟尺寸设置cuFFT计划，用于R2C（实到复）和C2R（复到实）变换

    // 预计算关联函数所需参数
    CalcCorrelationPara CorrP; // 关联函数参数结构体
    PreCalcCorrelations(InputC, InputP, CorrP); // 预计算关联函数所需的常数，这些常数在模拟过程中不变

    // 设置x、y、z方向的应变
    float eps[3] = { 0, 0, 0 }; // 初始化应变数组
    Strain(eps, 0, InputC, InputP, SimP); // 设置初始应变（这里设置为0，如果需要可随时间变化）

    // 初始化场变量n和g
    if(InputC.restartFlag) // 如果restartFlag为真，则从重启文件初始化
        restart(Var.n, Var.g, InputC, SimP, InputC.restartTime); // 从指定时间步的重启文件加载n和g
    else // 否则，进行随机或预设的初始化
        initialize <<< dimGrid3D, dimBlock3D >>> 
		(n_pack, g_pack, 1, InputC, InputP, SimP); // kernel初始化
    CheckLastCudaError();

    /*------取消注释此部分以输出初始状态------*/
    //cudaDeviceSynchronize(); // 等待所有CUDA操作完成，确保数据在输出前已准备好
    //output(Var.n, Var.g, InputC, SimP, 0); // 输出初始状态的n和g场到文件

    std::cout << "Loop start now..." << std::endl; // 打印提示信息，模拟循环即将开始

    /*-----------------------------模拟主循环---------------------------------------*/

    /*-------------------------性能测试-------------------------*/
    time_t t_start_loop, t_end_loop; // 用于记录循环运行时间的变量
    t_start_loop = clock();          // 记录循环开始时的CPU时间
    float kernel_time_1, kernel_time_2, kernel_time_3, kernel_time_4, 
		  fft_time_1, fft_time_2, fft_time_3, fft_time_4; // 记录每个核函数和FFT操作的时间
    float kernel_tot_1 = 0, kernel_tot_2 = 0, kernel_tot_3 = 0, kernel_tot_4 = 0, 
		  fft_tot_1 = 0, fft_tot_2 = 0, fft_tot_3 = 0, fft_tot_4 = 0; // 累积总时间
    cudaEvent_t kernel_start, kernel_end, fft_start, fft_end; // CUDA事件，精确测量GPU时间
    cudaEventCreate(&kernel_start); cudaEventCreate(&kernel_end);
    cudaEventCreate(&fft_start);cudaEventCreate(&fft_end);

    // 模拟主循环，从restartTime + 1开始，直到totalTime结束
    for (int t = InputC.restartTime + 1; t < InputC.totalTime; t++)
    {
        /*-----------如果应用了变化的应变，请取消注释此部分-------------*/
        //Set strain in x, y, z directions
        cudaDeviceSynchronize(); // 确保之前的操作完成，以避免时间测量干扰
        //double eps[3] = { 0, 0, 0 }; // 可以在这里根据时间步t设置变化的应变
        //Strain(eps, t, InputC, InputP, SimP);
        if (InputP.gamma13_switch) // 如果gamma13_switch为真，设置耦合项
            SetCouplingTerm(t, InputP, SimP); // 设置与时间相关的耦合项

        /*------------------------性能测试------------------------*/
        cudaEventRecord(kernel_start, 0); // 记录核函数开始时间

        // k空间计算关联函数 (omegak)
        CalcCorrelation <<< dimGrid3D, dimBlock3D >>> (karr_pack, omegak_pack, CorrP, SimP);

        /*------------------------性能测试------------------------*/
        cudaEventRecord(kernel_end, 0);    // 记录核函数结束时间
        cudaEventSynchronize(kernel_end);  // 等待核函数完成
        cudaEventElapsedTime(&kernel_time_1, kernel_start, kernel_end); // 计算核函数执行时间

        /*------------------------性能测试------------------------*/
        cudaEventRecord(fft_start, 0); // 记录FFT开始时间

        // 执行n到kn的实到复（R2C）FFT变换
        // 将实空间密度场n变换到k空间（复数形式）kn
        for (int p = 0; p < 3; p++) // 假设n是三维数组，或者有三个分量需要分别FFT
        { CHECK(cufftExecR2C(FFTH.planF_n[p], Var.n[p], Var.kn[p])); }// 执行FFT

        /*------------------------性能测试------------------------*/
        cudaEventRecord(fft_end, 0);    // 记录FFT结束时间
        cudaEventSynchronize(fft_end);  // 等待FFT完成
        cudaEventElapsedTime(&fft_time_1, fft_start, fft_end); // 计算FFT执行时间

        /*------------------------性能测试------------------------*/
        cudaEventRecord(kernel_start, 0); // 记录核函数开始时间

        // 在实空间计算非线性项nfNL
        // nfNL通常是n^3或n*n等非线性函数，在实空间计算更高效
        CalcNL <<<dimGrid3D, dimBlock3D >>> (n_pack, nfNL_pack, InputP, SimP);

        /*------------------------性能测试------------------------*/
        cudaEventRecord(kernel_end, 0);    // 记录核函数结束时间
        cudaEventSynchronize(kernel_end);  // 等待核函数完成
        cudaEventElapsedTime(&kernel_time_2, kernel_start, kernel_end); // 计算核函数执行时间

        /*------------------------性能测试------------------------*/
        cudaEventRecord(fft_start, 0); // 记录FFT开始时间

        // 执行nfNL到knfNL的实到复（R2C）FFT变换
        // 将实空间非线性项nfNL变换到k空间（复数形式）knfNL
        for (int p = 0; p < 3; p++)
        { CHECK(cufftExecR2C(FFTH.planF_NL[p], Var.nfNL[p], Var.knfNL[p])); } // 执行FFT

        /*------------------------性能测试------------------------*/
        cudaEventRecord(fft_end, 0);    // 记录FFT结束时间
        cudaEventSynchronize(fft_end);  // 等待FFT完成
        cudaEventElapsedTime(&fft_time_2, fft_start, fft_end); // 计算FFT执行时间

        // 根据是否启用iwave模式（可能是某种时间积分方案或特定物理模型）执行不同的更新逻辑
        if (InputC.iwave)
        {
            /*------------------------性能测试------------------------*/
            cudaEventRecord(fft_start, 0); // 记录FFT开始时间

            // 执行g到kg的实到复（R2C）FFT变换
            // 将辅助场变量g变换到k空间（复数形式）kg
            for (int p = 0; p < 3; p++)
            { CHECK(cufftExecR2C(FFTH.planF_g[p], Var.g[p], Var.kg[p])); }

            /*------------------------性能测试------------------------*/
            cudaEventRecord(fft_end, 0);    // 记录FFT结束时间
            cudaEventSynchronize(fft_end);  // 等待FFT完成
            cudaEventElapsedTime(&fft_time_3, fft_start, fft_end); // 计算FFT执行时间

            /*------------------------性能测试------------------------*/
            cudaEventRecord(kernel_start, 0); // 记录核函数开始时间

            // 在k空间更新kn (密度场n的傅里叶变换)
            // 这里的更新可能涉及到PFC方程的线性项、非线性项以及辅助场g的贡献
            UpdateKnWave <<<dimGrid3D, dimBlock3D >>> 
			(kn_pack, kg_pack, knfNL_pack, karr_pack, omegak_pack,InputC, InputP, SimP);

            // 第一次更新实空间中的g (g = n_old)
            // 这可能是为了存储当前时间步的n值，以便在下一个时间步计算g的更新
            UpdateG1 <<<dimGrid3D, dimBlock3D >>> (n_pack, g_pack, SimP);

            /*------------------------性能测试------------------------*/
            cudaEventRecord(kernel_end, 0);    // 记录核函数结束时间
            cudaEventSynchronize(kernel_end);  // 等待核函数完成
            cudaEventElapsedTime(&kernel_time_3, kernel_start, kernel_end); // 计算核函数执行时间

        }
        else // 如果不启用iwave模式
            // 在k空间更新kn (密度场n的傅里叶变换)
            // 这里的更新可能只涉及PFC方程的线性项和非线性项，不涉及辅助场g
            UpdateKn <<<dimGrid3D, dimBlock3D >>> (kn_pack, knfNL_pack, karr_pack, omegak_pack,
                InputC, InputP, SimP);

        /*------------------------性能测试------------------------*/
        cudaEventRecord(fft_start, 0); // 记录FFT开始时间

        // 执行kn到n的复到实（C2R）FFT变换
        for (int p = 0; p < 3; p++)
        {
            // 在C2R FFT之前进行预归一化，因为cuFFT的C2R变换通常不带归一化因子
            pre_normalize <<<dimGrid3D, dimBlock3D >>> (Var.kn[p], SimP);
            CHECK(cufftExecC2R(FFTH.planB_n[p], Var.kn[p], Var.n[p])); // 执行FFT
            // normalize <<< dimGrid1D, dimBlock1D >>> (Var.n[p], SimP); // 可能会有额外的归一化步骤，这里被注释掉了
        }

        /*------------------------性能测试------------------------*/
        cudaEventRecord(fft_end, 0);    // 记录FFT结束时间
        cudaEventSynchronize(fft_end);  // 等待FFT完成
        cudaEventElapsedTime(&fft_time_4, fft_start, fft_end); // 计算FFT执行时间

        /*------------------------性能测试------------------------*/
        cudaEventRecord(kernel_start, 0); // 记录核函数开始时间

        // 第二次更新实空间中的g [g = (n_new - n_old)/dt]
        // 这通常用于计算场变量n的时间导数，作为辅助场
        if(InputC.iwave) // 只有在iwave模式下才更新g
            UpdateG2 <<<dimGrid3D, dimBlock3D >>> (n_pack, g_pack, InputP, SimP);

        /*------------------------性能测试------------------------*/
        cudaEventRecord(kernel_end, 0);    // 记录核函数结束时间
        cudaEventSynchronize(kernel_end);  // 等待核函数完成
        cudaEventElapsedTime(&kernel_time_4, kernel_start, kernel_end); // 计算核函数执行时间
        
        CheckLastCudaError(); // 检查所有CUDA操作是否有错误

        // 写入输出文件
        if (t % InputC.printFreq == 0) // 每隔printFreq个时间步输出一次
        {
            cudaDeviceSynchronize(); // 等待所有CUDA操作完成，确保数据在输出前已准备好
            output(Var.n, Var.g, InputC, SimP, t); // 输出当前时间步的n和g场到文件
        }

        /*------------------------性能测试------------------------*/
        cudaDeviceSynchronize(); // 等待所有CUDA操作完成，确保时间测量准确
        // 累积每个核函数和FFT操作的总时间
        kernel_tot_1 += kernel_time_1; kernel_tot_2 += kernel_time_2; kernel_tot_3 += kernel_time_3; kernel_tot_4 += kernel_time_4;
        fft_tot_1 += fft_time_1; fft_tot_2 += fft_time_2; fft_tot_3 += fft_time_3; fft_tot_4 += fft_time_4;

    } // 模拟循环结束

    cudaDeviceSynchronize(); // 确保所有GPU操作在程序退出前完成

    /*------------------------性能测试------------------------*/
    t_end_loop = clock(); // 记录循环结束时的CPU时间
    // 销毁CUDA事件，释放资源
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(fft_start);
    cudaEventDestroy(fft_end);

    DestroyCufftPlan(FFTH);     // 销毁cuFFT计划，释放相关资源
    FreeMemory(Var, InputC);    // 释放Var中分配的GPU（管理）内存
    CheckLastCudaError();       // 再次检查是否有CUDA错误

    /*------------------------性能测试------------------------*/
    t_end_tot = clock(); // 记录程序总结束时的CPU时间
    // 打印各种性能统计信息
    printf("The overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);
    printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop)/(double)(t_end_tot - t_start_tot)*100.);
    printf("The kernels running time is: %f sec.\n", ((float)(kernel_tot_1 + kernel_tot_2 + kernel_tot_3 + kernel_tot_4)) / 1000.0); // 注意这里除以1000.0是因为cudaEventElapsedTime返回的是毫秒
    printf("In detail: kernel1 - %f sec, kernel2 - %f sec, kernel3 - %f sec, kernel4 - %f sec.", (float)(kernel_tot_1) / 1000.0, (float)(kernel_tot_2) / 1000.0, (float)(kernel_tot_3) / 1000.0, (float)(kernel_tot_4) / 1000.0);
    printf("The cufft running time is: %f sec.\n", ((float)(fft_tot_1 + fft_tot_2 + fft_tot_3 + fft_tot_4)) / 1000.0); // 注意这里除以1000.0是因为cudaEventElapsedTime返回的是毫秒
    printf("In detail: fft1 - %f sec, fft2 - %f sec, fft3 - %f sec, fft4 - %f sec.", (float)(fft_tot_1) / 1000.0, (float)(fft_tot_2) / 1000.0, (float)(fft_tot_3) / 1000.0, (float)(fft_tot_4) / 1000.0);

}
