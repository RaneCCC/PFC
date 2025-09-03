/*
2025.8 v6 PFC-RG晶粒生长 cuda版
(基于2006 Renormalization Group方法 DOI: 10.1007/s10955-005-9013-7) 
1.使用gpu线程索引计算宏 用来优化合并内存访问 使每个线程进行跨block计算
2.去除指向A_R数据的h_A_R指针，替换为vector.data()
3.仅在每freq输出的时候进行icufft、归一化和 gpu返回cpu数据，减少异端通信
4.main函数开始可以自行设置block以及grid的gpu参数设置
5.macro：使用宏打包A_R和nl_R数组 kernel更简洁一些
6.添加了cudaevent用于计时，包括初始化 传输数据 fft ifft 归一化等过程的计时
7.macro：添加了错误检查宏 CUDA_CHECK
8.
nvcc -O3 -o apfc_cuda apfc_cuda.cu -lcufft
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <complex>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace std;
using z_H = complex<double>;
using z_D = cuDoubleComplex;
using cufft_z = cufftDoubleComplex;// 对应的cufftComplex 是float


/*-------------------------------- macro --------------------------------*/
// cuda错误检查宏 cudaMalloc cudaMemcpy cudaEvent...
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cufft错误检查宏 cufftPlan2d cufftExecZ2Z cufftDestroy...
#define CUFFT_CHECK(call) do { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT错误 %s:%d: 错误代码 %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// gpu线程索引计算宏
#define THREAD_LOOP_GPU(size) \
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x); \
    int stride = gridDim.x * gridDim.y * blockDim.x * blockDim.y; \
    for (int idx = tid; idx < size; idx += stride)
/*--------------------------------- macro --------------------------------*/



/*----------------------- 点乘 inline ------------------------*/
__device__ inline double dot_D(z_D k, z_D x)
   //cuCreal() : 提取复数实部 ; cuCimag() : 提取复数虚部
   { return cuCreal(k) * cuCreal(x) + cuCimag(k) * cuCimag(x); }

inline double dot_H(z_H k, z_H x)
    { return k.real() * x.real() + k.imag() * x.imag(); }


/*------------------------ class ---------------------------*/
class Input_Para{// 输入参数
public:
    int N;
    double dx;
    double dt; // 时间步长
    int steps; // 总步数
    int freq; // 输出频率

    double psi_c;
    double r;
    double k0;
    double A0;
    
    // constructor
    Input_Para(): N(256), dx(M_PI/2.), dt(0.04), steps(10000), freq(1000), 
                  psi_c(0.285), r(-0.25), k0(1.), A0(0.1){}

    void print(){
        cout << "======== 输入参数 =======" << endl;
        cout << "N = " << N << endl;
        cout << "dx = " << dx << endl;
        cout << "dt = " << dt << endl;
        cout << "steps = " << steps << endl;
        cout << "freq = " << freq << endl;
        cout << "psi_c = " << psi_c << endl;
        cout << "r = " << r << endl;
        cout << "k0 = " << k0 << endl;
        cout << "A0 = " << A0 << endl;

    }
};

class PreCal_Para{// 预计算参数 包括Szie 和一些算子
public:
    int Size;
    double Gamma;// 扰动增长率计算的常数计算Γ
    double factor_k;// 基本波矢单位
    double psi_c_6;

    vector<z_H> k_vec;// 后期可改成fcc bcc

    PreCal_Para(const Input_Para& params){
        Size = params.N * params.N;
        Gamma = -(params.r + 3 * params.psi_c * params.psi_c);
        factor_k = 2.0 * M_PI / (params.N * params.dx);
        psi_c_6 = 6.0 * params.psi_c;
    }
};

class Fields{// 场变量 主要是A和nl
public:
    vector<z_H> A_R[3], nl_R[3];// cpu 复数域数据
    z_D *h_A_R[3], *h_nl_R[3];// cpu 复数域数据指针
    z_D *d_A_R[3], *d_nl_R[3];// gpu 复数域数据指针
    cufft_z *d_A_k[3], *d_nl_k[3];// gpu 频域数据指针
    cufftHandle plan_forward, plan_backward;

    // constructor 初始化
    Fields(const Input_Para& params, const PreCal_Para& pre)
    {   // 初始化cpu数据（也可以在{}外单独设置vector<z_H>）
        for(int i = 0; i < 3; ++i){
            A_R[i].resize(pre.Size);
            nl_R[i].resize(pre.Size);
            h_A_R[i] = reinterpret_cast<z_D*>(A_R[i].data());// 确保z_H和z_D内存布局一致
            h_nl_R[i] = reinterpret_cast<z_D*>(nl_R[i].data());
        }
        
        // 初始化gpu内存
        for(int i = 0; i < 3; ++i){
            CUDA_CHECK(cudaMalloc(&d_A_R[i], sizeof(z_D) * pre.Size));
            CUDA_CHECK(cudaMalloc(&d_nl_R[i], sizeof(z_D) * pre.Size));
            CUDA_CHECK(cudaMalloc(&d_A_k[i], sizeof(z_D) * pre.Size));
            CUDA_CHECK(cudaMalloc(&d_nl_k[i], sizeof(z_D) * pre.Size));
        }

        // 初始化cufft Plan 
        // (计划指针, 宽度, 高度, 变换类型) Z2Z 双精度复数到复数的变换（Z=double complex）
        CUFFT_CHECK(cufftPlan2d(&plan_forward, params.N, params.N, CUFFT_Z2Z));
        CUFFT_CHECK(cufftPlan2d(&plan_backward, params.N, params.N, CUFFT_Z2Z));
    }

    // 数据传输 HtoD 
    void cpMemHtoD(const PreCal_Para& pre){
        for(int i = 0; i < 3; ++i){
            CUDA_CHECK(cudaMemcpy(d_A_R[i], h_A_R[i], sizeof(z_D) * pre.Size, cudaMemcpyHostToDevice));}
    }

    // 数据传输 DtoH 
    void cpMemDtoH(const PreCal_Para& pre){
        for(int i = 0; i < 3; ++i){
            CUDA_CHECK(cudaMemcpy(h_A_R[i], d_A_R[i], sizeof(z_D) * pre.Size, cudaMemcpyDeviceToHost));}
    }

    ~Fields(){
        for(int i = 0; i < 3; ++i){
            CUDA_CHECK(cudaFree(d_A_R[i]));
            CUDA_CHECK(cudaFree(d_nl_R[i]));
            CUDA_CHECK(cudaFree(d_A_k[i]));
            CUDA_CHECK(cudaFree(d_nl_k[i]));
        }
        CUFFT_CHECK(cufftDestroy(plan_forward));
        CUFFT_CHECK(cufftDestroy(plan_backward));
    }
};

class cudaTimer {// cuda计时用于性能测试
public:
    cudaEvent_t start_init, stop_init;
    cudaEvent_t start_HtoD, stop_HtoD;
    cudaEvent_t start_nl, stop_nl;
    cudaEvent_t start_fft, stop_fft;
    cudaEvent_t start_ifft, stop_ifft;
    cudaEvent_t start_update, stop_update;
    cudaEvent_t start_DtoH, stop_DtoH;
    
    float t_init = 0.0f, t_HtoD = 0.0f, t_nl = 0.0f, t_fft = 0.0f;
    float t_ifft = 0.0f, t_update = 0.0f, t_DtoH = 0.0f;
    float elapsed_init, elapsed_HtoD, elapsed_nl, elapsed_fft;
    float elapsed_ifft, elapsed_update, elapsed_DtoH;
    
    // 构造函数：创建所有事件
    cudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_init));CUDA_CHECK(cudaEventCreate(&stop_init));
        CUDA_CHECK(cudaEventCreate(&start_HtoD));CUDA_CHECK(cudaEventCreate(&stop_HtoD));
        CUDA_CHECK(cudaEventCreate(&start_nl));CUDA_CHECK(cudaEventCreate(&stop_nl));
        CUDA_CHECK(cudaEventCreate(&start_fft));CUDA_CHECK(cudaEventCreate(&stop_fft));
        CUDA_CHECK(cudaEventCreate(&start_ifft));CUDA_CHECK(cudaEventCreate(&stop_ifft));
        CUDA_CHECK(cudaEventCreate(&start_update));CUDA_CHECK(cudaEventCreate(&stop_update));
        CUDA_CHECK(cudaEventCreate(&start_DtoH));CUDA_CHECK(cudaEventCreate(&stop_DtoH));
    }
    
    // 析构函数：销毁所有事件
    ~cudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_init));CUDA_CHECK(cudaEventDestroy(stop_init));
        CUDA_CHECK(cudaEventDestroy(start_HtoD));CUDA_CHECK(cudaEventDestroy(stop_HtoD));
        CUDA_CHECK(cudaEventDestroy(start_nl));CUDA_CHECK(cudaEventDestroy(stop_nl));
        CUDA_CHECK(cudaEventDestroy(start_fft));CUDA_CHECK(cudaEventDestroy(stop_fft));
        CUDA_CHECK(cudaEventDestroy(start_ifft));CUDA_CHECK(cudaEventDestroy(stop_ifft));
        CUDA_CHECK(cudaEventDestroy(start_update));CUDA_CHECK(cudaEventDestroy(stop_update));
        CUDA_CHECK(cudaEventDestroy(start_DtoH));CUDA_CHECK(cudaEventDestroy(stop_DtoH));
    }
};


/*-------------------------- 功能性函数 -----------------------------*/
// 从配置文件读取参数
void readInput(const string& filename, Input_Para& params) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cout << "无法打开配置文件: " << filename << "，使用默认参数" << endl;
        return; // 使用默认参数，不退出程序
    }
    
    string line;
    getline(infile, line); stringstream(line) >> params.N;
    getline(infile, line); stringstream(line) >> params.dx;
    getline(infile, line); stringstream(line) >> params.dt;
    getline(infile, line); stringstream(line) >> params.steps;
    getline(infile, line); stringstream(line) >> params.freq;
    getline(infile, line); stringstream(line) >> params.psi_c;
    getline(infile, line); stringstream(line) >> params.r;
    getline(infile, line); stringstream(line) >> params.k0;
    getline(infile, line); stringstream(line) >> params.A0;

    infile.close();
    cout << "Input read from " << filename << endl;
}

// 创建输出文件夹 返回输出路径
string get_output_path(const string& filename) {
    string folder = "output"; // 创建output文件夹
    if (!filesystem::exists(folder)) { filesystem::create_directory(folder); }
    return folder + "/" + filename;
}

// 输出vtk文件 (几乎不变 蛋接收重构后的psi)
void output_vtk(const vector<double>& psi, int N, const string& filename) {
    string full_path = get_output_path(filename);
    ofstream fout_vtk(full_path);
    
    fout_vtk << "# vtk DataFile Version 3.0\n";
    fout_vtk << "PFC-RG simulation results\n";
    fout_vtk << "ASCII\n";
    fout_vtk << "DATASET STRUCTURED_POINTS\n";
    fout_vtk << "DIMENSIONS " << N << " " << N << " 1\n";
    fout_vtk << "ORIGIN 0 0 0\n";
    fout_vtk << "SPACING 1 1 1\n";
    fout_vtk << "POINT_DATA " << N * N << "\n";
    fout_vtk << "SCALARS psi double\nLOOKUP_TABLE default\n";
    
    for (int i = 0; i < N * N; ++i) { fout_vtk << psi[i] << "\n"; }

    fout_vtk.close();
    cout << "Output file: " << filename << endl;
}


/*------------------------------ 主要函数 -----------------------------------*/
// gpu 计算非线性项 
__global__ void compute_nl_kernel(z_D* A0_R, z_D* A1_R, z_D* A2_R, z_D* nl0_R, z_D* nl1_R, z_D* nl2_R, int Size, double psi_c_6) {
    THREAD_LOOP_GPU(Size) {// 优化合并内存访问
        // 计算模长平方 
        double A_abs2[3] = { dot_D(A0_R[idx], A0_R[idx]),// A0_R是指针，A0_R[idx]是通过指针访问的具体值 类型为z_D
                             dot_D(A1_R[idx], A1_R[idx]),
                             dot_D(A2_R[idx], A2_R[idx]) };
        
        // 计算共轭
        z_D A_conj[3] = { make_cuDoubleComplex(cuCreal(A0_R[idx]), -cuCimag(A0_R[idx])),
                          make_cuDoubleComplex(cuCreal(A1_R[idx]), -cuCimag(A1_R[idx])),
                          make_cuDoubleComplex(cuCreal(A2_R[idx]), -cuCimag(A2_R[idx])) };
        
        // 计算非线性项
        z_D term1_0 = cuCmul( make_cuDoubleComplex(-3.0 * (A_abs2[0] + 2.0 * (A_abs2[1] + A_abs2[2])), 0), A0_R[idx] );
        z_D term2_0 = cuCmul( make_cuDoubleComplex(-psi_c_6, 0), cuCmul(A_conj[1], A_conj[2]) );
        nl0_R[idx]  = cuCadd( term1_0, term2_0 );
        
        z_D term1_1 = cuCmul( make_cuDoubleComplex(-3.0 * (A_abs2[1] + 2.0 * (A_abs2[0] + A_abs2[2])), 0), A1_R[idx] );
        z_D term2_1 = cuCmul( make_cuDoubleComplex(-psi_c_6, 0), cuCmul(A_conj[0], A_conj[2]) );
        nl1_R[idx]  = cuCadd( term1_1, term2_1 );
        
        z_D term1_2 = cuCmul( make_cuDoubleComplex(-3.0 * (A_abs2[2] + 2.0 * (A_abs2[0] + A_abs2[1])), 0), A2_R[idx] );
        z_D term2_2 = cuCmul( make_cuDoubleComplex(-psi_c_6, 0), cuCmul(A_conj[0], A_conj[1]) );
        nl2_R[idx]  = cuCadd( term1_2, term2_2 );
    } 
}

// gpu 归一化
__global__ void normalize_kernel(z_D* A0_R, z_D* A1_R, z_D* A2_R, int Size) {
    double norm_factor = 1.0 / static_cast<double>(Size);
    z_D norm_complex = make_cuDoubleComplex(norm_factor, 0);
    
    THREAD_LOOP_GPU(Size) {// 优化合并内存访问 处理多个元素以提高内存带宽利用率
        A0_R[idx] = cuCmul( A0_R[idx], norm_complex );
        A1_R[idx] = cuCmul( A1_R[idx], norm_complex );
        A2_R[idx] = cuCmul( A2_R[idx], norm_complex );
    }
}

// gpu 频域更新
__global__ void update_k_kernel(cufft_z* A_k, const cufft_z* nl_k, z_D k_j, int Size, 
                                int N, double dt, double factor_k, double Gamma) {
    THREAD_LOOP_GPU(Size) {
        int i = idx / N;// 从线性索引计算2D坐标
        int j = idx % N;
        
        double ky = (i < N / 2) ? i * factor_k : (i - N) * factor_k;// 计算波矢
        double kx = (j < N / 2) ? j * factor_k : (j - N) * factor_k;

        z_D q0 = make_cuDoubleComplex(kx, ky);// 将C++的double值转换为CUDA复数类型
        z_D q1 = cuCadd(q0, k_j);// 经过晶粒波矢k_j移位后的波矢
        double op_2 = -(dot_D(q0, q0) + 2 * dot_D(k_j, q0));// 算子平方项

        // 旋转协变 线性算子Lj
        double Lj = dot_D(q1, q1) * (Gamma - op_2 * op_2);

        // 时间步进 半隐式欧拉法
        z_D Aj = A_k[idx];
        z_D nl = nl_k[idx];
        z_D dt_nl = cuCmul( make_cuDoubleComplex(dt, 0), nl );// dt*nl
        z_D fac1 = cuCadd( Aj, dt_nl );// Aj + dt*nl_k
        double fac2 = 1.0 - dt * Lj;
        
        A_k[idx] = cuCdiv( fac1, make_cuDoubleComplex(fac2, 0) );//(Aj + dt * nl_k) / (1. - dt * Lj);
    } // 结束for循环
}

// cpu 初始化
void initialize(const Input_Para& params, PreCal_Para& pre, Fields& fields) {
    // 初始化1 倒格矢
    z_H k1 = z_H( -sqrt(3.0) / 2.0, -0.5 );
    z_H k2 = z_H( 0.0, 1.0 );
    z_H k3 = z_H( sqrt(3.0) / 2.0, -0.5 );
    pre.k_vec = { k1, k2, k3 };
    
    // 初始化2 种子
    struct Seed { int x, y; double theta; double radius; };
    vector<Seed> seeds = {
        {   params.N/4,     params.N/4,      0,     15 },
        { 3*params.N/4,     params.N/4,    M_PI/6,  15 },
        {   params.N/2,   3*params.N/4,    M_PI/3,  15 }};
    
    for (int i = 0; i < params.N; ++i) {
        for (int j = 0; j < params.N; ++j) {
            int idx = i * params.N + j;
            fields.A_R[0][idx] = fields.A_R[1][idx] = fields.A_R[2][idx] = z_H(0.0, 0.0); // 默认液相
            z_H x_vec = z_H(j * params.dx, i * params.dx); // 网格坐标

            // 判断是否在晶粒内部 半径内激活初始波形 
            for (const Seed& s : seeds) {
                double dx_s = (j - s.x) * params.dx;// 到晶粒中心距离
                double dy_s = (i - s.y) * params.dx;
                double dist = sqrt(dx_s * dx_s + dy_s * dy_s);
                
                if (dist < s.radius) {
                    double cos_theta = cos(s.theta);
                    double sin_theta = sin(s.theta);
                    z_H k1_rot = z_H(k1.real() * cos_theta - k1.imag() * sin_theta,
                                     k1.real() * sin_theta + k1.imag() * cos_theta);
                    z_H k2_rot = z_H(k2.real() * cos_theta - k2.imag() * sin_theta,
                                     k2.real() * sin_theta + k2.imag() * cos_theta);
                    z_H k3_rot = z_H(k3.real() * cos_theta - k3.imag() * sin_theta,
                                     k3.real() * sin_theta + k3.imag() * cos_theta);

                    fields.A_R[0][idx] = params.A0 * exp( z_H(0, dot_H(k1_rot - k1, x_vec)) );
                    fields.A_R[1][idx] = params.A0 * exp( z_H(0, dot_H(k2_rot - k2, x_vec)) );
                    fields.A_R[2][idx] = params.A0 * exp( z_H(0, dot_H(k3_rot - k3, x_vec)) );
                }
            }
        }
    }
}

// cpu 重构psi 输出vtk
void reconstruct_output(int step, const Input_Para& params, const PreCal_Para& pre, const Fields& fields) {
    vector<double> psi_A(pre.Size);
    for (int k = 0; k < pre.Size; ++k) {
        int i = k / params.N;
        int j = k % params.N;
        z_H x_vec = z_H(j * params.dx, i * params.dx); // 网格坐标
        z_H A_sum = fields.A_R[0][k] * exp( z_H(0, dot_H(pre.k_vec[0], x_vec)) ) +
                    fields.A_R[1][k] * exp( z_H(0, dot_H(pre.k_vec[1], x_vec)) ) +
                    fields.A_R[2][k] * exp( z_H(0, dot_H(pre.k_vec[2], x_vec)) );
        psi_A[k] = params.psi_c + 2.0 * A_sum.real();
    }
    // 只有每freq步会把线性项和非线性项合并并转化为psi 并输出
    output_vtk(psi_A, params.N, "output_" + to_string(step) + ".vtk");
}



int main(){
    // 相关类对象及内存创建 cudaEvent开始计时 读取输入
    cudaTimer t_cuda;  // 创建和管理所有cudaEvent类 每次性能测试包括5行代码 含kernel的6行多cudaGetLastError()
    CUDA_CHECK(cudaEventRecord(t_cuda.start_init));// 记录开始时间
    Input_Para params;
    readInput("input.txt", params); 
    params.print();
    PreCal_Para pre(params);
    Fields fields(params, pre);

    // 初始化
    initialize(params, pre, fields);
    CUDA_CHECK(cudaEventRecord(t_cuda.stop_init));// 记录结束时间
    CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_init));// 等待kernel完成
    CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_init, t_cuda.start_init, t_cuda.stop_init));// 计算执行时间
    t_cuda.t_init = t_cuda.elapsed_init;

    // 根据网格大小N设置gpu启动参数 并打印
    dim3 dimBlock3D, dimGrid3D;
    if (params.N <= 128) {     // 256线程/块 64个线程块，总线程数16384
        dimBlock3D = dim3(16, 16, 1); dimGrid3D = dim3(8, 8, 1);} 
    else if (params.N <= 256) { // 256线程/块 256个线程块，总线程数65536
        dimBlock3D = dim3(16, 16, 1); dimGrid3D = dim3(16, 16, 1);} 
    else if (params.N <= 512) { // 256线程/块 1024个线程块，总线程数262144
        dimBlock3D = dim3(16, 16, 1); dimGrid3D = dim3(32, 32, 1);} 
    else if (params.N <= 1024) { // 1024线程/块 1024个线程块，总线程数1048576
        dimBlock3D = dim3(32, 32, 1); dimGrid3D = dim3(32, 32, 1);} 
    else {         // N = 2048   // 1024线程/块 4096个线程块，总线程数4194304
        dimBlock3D = dim3(32, 32, 1); dimGrid3D = dim3(64, 64, 1);} 
    printf("GPU配置: Block(%d,%d,%d), Grid(%d,%d,%d)\n", 
            dimBlock3D.x, dimBlock3D.y, dimBlock3D.z, dimGrid3D.x, dimGrid3D.y, dimGrid3D.z);
    printf("总线程数: %d, 数据大小: %d\n=================================\n", 
            dimBlock3D.x * dimBlock3D.y * dimGrid3D.x * dimGrid3D.y, params.N * params.N);

    /*----------------- 初始数据传输到gpu ------------------*/
    CUDA_CHECK(cudaEventRecord(t_cuda.start_HtoD));
    fields.cpMemHtoD(pre);
    CUDA_CHECK(cudaEventRecord(t_cuda.stop_HtoD));// 记录结束时间
    CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_HtoD));// 等待kernel完成
    CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_HtoD, t_cuda.start_HtoD, t_cuda.stop_HtoD));// 计算执行时间
    t_cuda.t_HtoD = t_cuda.elapsed_HtoD;


    // gpu 开始模拟
    reconstruct_output(0, params, pre, fields);
    for (int t = 1; t < params.steps + 1; ++t) {
        /*------------------ 1.gpu计算非线性项 ----------------------*/
        CUDA_CHECK(cudaEventRecord(t_cuda.start_nl));
        compute_nl_kernel<<< dimGrid3D, dimBlock3D >>>
            (fields.d_A_R[0], fields.d_A_R[1], fields.d_A_R[2], 
             fields.d_nl_R[0], fields.d_nl_R[1], fields.d_nl_R[2], pre.Size, pre.psi_c_6);
        CUDA_CHECK(cudaGetLastError()); // 检查kernel启动错误
        CUDA_CHECK(cudaEventRecord(t_cuda.stop_nl));
        CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_nl));
        CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_nl, t_cuda.start_nl, t_cuda.stop_nl));
        t_cuda.t_nl += t_cuda.elapsed_nl;

        /*------------------ 2.fft 转到频域 ----------------------*/
        CUDA_CHECK(cudaEventRecord(t_cuda.start_fft));
        for (int j = 0; j < 3; ++j) {
            // cufft
            CUFFT_CHECK(cufftExecZ2Z(fields.plan_forward, (cufft_z*)fields.d_A_R[j], fields.d_A_k[j], CUFFT_FORWARD));
            CUFFT_CHECK(cufftExecZ2Z(fields.plan_forward, (cufft_z*)fields.d_nl_R[j], fields.d_nl_k[j], CUFFT_FORWARD));
        }
        CUDA_CHECK(cudaEventRecord(t_cuda.stop_fft));
        CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_fft));
        CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_fft, t_cuda.start_fft, t_cuda.stop_fft));
        t_cuda.t_fft += t_cuda.elapsed_fft;
        
        /*------------------ 3.频域更新 ----------------------*/
        CUDA_CHECK(cudaEventRecord(t_cuda.start_update));
        for (int j = 0; j < 3; ++j) {
            z_D k_j = make_cuDoubleComplex(pre.k_vec[j].real(), pre.k_vec[j].imag());
            update_k_kernel<<< dimGrid3D, dimBlock3D >>>
                (fields.d_A_k[j], fields.d_nl_k[j], k_j, pre.Size, params.N, params.dt, pre.factor_k, pre.Gamma);
        }
        CUDA_CHECK(cudaGetLastError()); // 检查kernel启动错误
        CUDA_CHECK(cudaEventRecord(t_cuda.stop_update));
        CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_update));
        CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_update, t_cuda.start_update, t_cuda.stop_update));
        t_cuda.t_update += t_cuda.elapsed_update;
        
        /*-------------------- 4.ifft + normallize ---------------------*/
        CUDA_CHECK(cudaEventRecord(t_cuda.start_ifft));
        for (int j = 0; j < 3; ++j) {
            CUFFT_CHECK(cufftExecZ2Z(fields.plan_backward, fields.d_A_k[j], (cufft_z*)fields.d_A_R[j], CUFFT_INVERSE)); }
        normalize_kernel<<< dimGrid3D, dimBlock3D >>> (fields.d_A_R[0], fields.d_A_R[1], fields.d_A_R[2], pre.Size);
        CUDA_CHECK(cudaEventRecord(t_cuda.stop_ifft));
        CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_ifft));
        CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_ifft, t_cuda.start_ifft, t_cuda.stop_ifft));
        t_cuda.t_ifft += t_cuda.elapsed_ifft;

        /*------------------ 5.按freq 重构psi输出 ---------------------*/
        if (t % params.freq == 0 || t == 0) {// 只在输出时传输数据到CPU
            /*---------------- 传输数据到cpu并输出 -----------------*/
            CUDA_CHECK(cudaEventRecord(t_cuda.start_DtoH));
            fields.cpMemDtoH(pre);
            reconstruct_output(t, params, pre, fields);// 不含kernel
            CUDA_CHECK(cudaEventRecord(t_cuda.stop_DtoH));
            CUDA_CHECK(cudaEventSynchronize(t_cuda.stop_DtoH));
            CUDA_CHECK(cudaEventElapsedTime(&t_cuda.elapsed_DtoH, t_cuda.start_DtoH, t_cuda.stop_DtoH));
            t_cuda.t_DtoH += t_cuda.elapsed_DtoH;
        }
    }  

    printf("\n================= 性能统计 =================\n");
    printf("初始化:   | %8.2f ms\n", t_cuda.t_init);
    printf("HtoD:     | %8.2f ms\n", t_cuda.t_HtoD);
    printf("nl_R:     | %8.2f ms   (平均: %.2f ms/步)\n", t_cuda.t_nl, t_cuda.t_nl/(params.steps + 1));
    printf("fft:      | %8.2f ms   (平均: %.2f ms/步)\n", t_cuda.t_fft, t_cuda.t_fft/(params.steps + 1));
    printf("ifft:     | %8.2f ms   (平均: %.2f ms/步)\n", t_cuda.t_ifft, t_cuda.t_ifft/(params.steps + 1));
    printf("update_k: | %8.2f ms   (平均: %.2f ms/步)\n", t_cuda.t_update, t_cuda.t_update/(params.steps + 1));
    printf("DtoH:     | %8.2f ms\n", t_cuda.t_DtoH);
    printf("总时间:   | %8.2f ms\n", t_cuda.t_init + t_cuda.t_HtoD + t_cuda.t_nl + t_cuda.t_fft + t_cuda.t_ifft + t_cuda.t_update + t_cuda.t_DtoH);
    printf("模拟完成！\n");
    return 0;
}