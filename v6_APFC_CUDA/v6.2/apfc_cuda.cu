/*
2025.8 v6.2 PFC-RG晶粒生长 cuda版
(基于2006 Renormalization Group方法 DOI: 10.1007/s10955-005-9013-7) 

相较于v6.1：
1.去掉全局参数类对象的创建，换成封装在类Input_Para里的
2.移除了中间层h指针h_A/nl_R[3] -> 直接使用vector.data() 去除reinterpret_cast强制类型转换
3.cpMemHtoD/DtoH只传输A_R，不传输nl_R（因为nl_R在GPU上计算）
4. 添加A0参数到Input_Para

nvcc -O3 -o apfc_cuda v6.2.cu -lcufft
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

__device__ inline double dot_D(z_D k, z_D x)
   //cuCreal() : 提取复数实部 ; cuCimag() : 提取复数虚部
   { return cuCreal(k) * cuCreal(x) + cuCimag(k) * cuCimag(x); }

inline double dot_H(z_H k, z_H x)
    { return k.real() * x.real() + k.imag() * x.imag(); }

/*-------------------------- class -----------------------------*/
class Input_Para{
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
                  psi_c(0.285), r(-0.25), k0(1.0), A0(0.1){}

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

class PreCal_Para{
public:
    int Size;
    double Gamma;// 扰动增长率计算的常数计算Γ
    double factor_k;// 基本波矢单位
    double psi_c_6;

    z_H k1, k2, k3;
    vector<z_H> k_vec;

    PreCal_Para(const Input_Para& params){
        Size = params.N * params.N;
        Gamma = -(params.r + 3 * params.psi_c * params.psi_c);
        factor_k = 2.0 * M_PI / (params.N * params.dx);
        psi_c_6 = 6.0 * params.psi_c;
    }
};

class Fields{
public:
    vector<z_H> A_R[3], nl_R[3];// cpu 复数域数据
    // cpu 复数域指针直接使用vector.data() 
    z_D *d_A_R[3], *d_nl_R[3];// gpu 复数域数据指针
    cufft_z *d_A_k[3], *d_nl_k[3];// gpu 频域数据指针
    cufftHandle plan_forward, plan_backward;

    // constructor 初始化
    Fields(const Input_Para& params, const PreCal_Para& pre)
    {   // 初始化cpu数据（也可以在{}外单独设置vector<z_H>）
        for(int i = 0; i < 3; ++i){
            A_R[i].resize(pre.Size);
            nl_R[i].resize(pre.Size);
        }
        
        // 初始化gpu内存
        for(int i = 0; i < 3; ++i){
            cudaMalloc(&d_A_R[i], sizeof(z_D) * pre.Size);
            cudaMalloc(&d_nl_R[i], sizeof(z_D) * pre.Size);
            cudaMalloc(&d_A_k[i], sizeof(z_D) * pre.Size);
            cudaMalloc(&d_nl_k[i], sizeof(z_D) * pre.Size);
        }

        // 初始化cufft Plan 
        // (计划指针, 宽度, 高度, 变换类型) Z2Z 双精度复数到复数的变换（Z=double complex）
        cufftPlan2d(&plan_forward, params.N, params.N, CUFFT_Z2Z);
        cufftPlan2d(&plan_backward, params.N, params.N, CUFFT_Z2Z);
    }

    // 数据传输 HtoD 
    void cpMemHtoD(const PreCal_Para& pre){
        for(int i = 0; i < 3; ++i){
            cudaMemcpy(d_A_R[i], A_R[i].data(), sizeof(z_D) * pre.Size, cudaMemcpyHostToDevice);}
    }

    // 数据传输 DtoH 
    void cpMemDtoH(const PreCal_Para& pre){
        for(int i = 0; i < 3; ++i){
            cudaMemcpy(A_R[i].data(), d_A_R[i], sizeof(z_D) * pre.Size, cudaMemcpyDeviceToHost);}
    }

    ~Fields(){
        for(int i = 0; i < 3; ++i){
            cudaFree(d_A_R[i]);
            cudaFree(d_nl_R[i]);
            cudaFree(d_A_k[i]);
            cudaFree(d_nl_k[i]);
        }
        cufftDestroy(plan_forward);
        cufftDestroy(plan_backward);
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
    fout_vtk << "SCALARS psi double\nlOOKUP_TABLE default\n";
    
    for (int i = 0; i < N * N; ++i) { fout_vtk << psi[i] << "\n"; }

    fout_vtk.close();
    cout << "Output file: " << filename << endl;
}



/*------------------------------ 主要函数 -----------------------------------*/
/*------------------------ gpu线程索引计算宏 -----------------------*/
#define GPU_THREAD_LOOP(size) \
    int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    int stride = blockDim.x * gridDim.x; \
    for (int idx = tid; idx < size; idx += stride)
/*------------------------ gpu线程索引计算宏 -----------------------*/

// gpu 计算非线性项 
__global__ void compute_nl_kernel(z_D* A0_R, z_D* A1_R, z_D* A2_R, z_D* nl0_R, z_D* nl1_R, z_D* nl2_R, int Size, double psi_c_6) {
    GPU_THREAD_LOOP(Size) {// 优化合并内存访问
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
    
    GPU_THREAD_LOOP(Size) {// 优化合并内存访问 处理多个元素以提高内存带宽利用率
        A0_R[idx] = cuCmul( A0_R[idx], norm_complex );
        A1_R[idx] = cuCmul( A1_R[idx], norm_complex );
        A2_R[idx] = cuCmul( A2_R[idx], norm_complex );
    }
}

// gpu 频域更新
__global__ void update_k_kernel(cufft_z* A_k, const cufft_z* nl_k, z_D k_j, int Size, 
                                int N, double dt, double factor_k, double Gamma) {
    GPU_THREAD_LOOP(Size) {
        int i = idx / N;// 从线性索引计算2D坐标
        int j = idx % N;
        
        double ky = (i < N / 2) ? i * factor_k : (i - N) * factor_k;// 计算波矢
        double kx = (j < N / 2) ? j * factor_k : (j - N) * factor_k;

        z_D q0 = make_cuDoubleComplex(kx, ky);// 将C++的double值转换为CUDA复数类型
        z_D q1 = cuCadd(q0, k_j);// 经过晶粒波矢k_j移位后的波矢
        double op_2 = -(dot_D(q0, q0) + 2 * dot_D(k_j, q0));// 算子平方项

        // 旋转协变 线性算子Lj
        double Lj = (dot_D(q1, q1)) * (Gamma - op_2 * op_2);

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
void initialize(const Input_Para& params, PreCal_Para*& pre_p, Fields*& fields_p) {
    // 初始化 预计算参数 fft 场变量指针
    pre_p = new PreCal_Para(params);
    fields_p = new Fields(params, *pre_p);

    // 初始化1 倒格矢
    z_H k1 = z_H( -sqrt(3.0) / 2.0, -0.5 );
    z_H k2 = z_H( 0.0, 1.0 );
    z_H k3 = z_H( sqrt(3.0) / 2.0, -0.5 );
    pre_p->k_vec = { k1, k2, k3 };
    
    // 初始化2 种子
    struct Seed { int x, y; double theta; double radius; };
    vector<Seed> seeds = {
        {   params.N/4,     params.N/4,      0,     15 },
        { 3*params.N/4,     params.N/4,    M_PI/6,  15 },
        {   params.N/2,   3*params.N/4,    M_PI/3,  15 }};
    
    for (int i = 0; i < params.N; ++i) {
        for (int j = 0; j < params.N; ++j) {
            int idx = i * params.N + j;
            fields_p->A_R[0][idx] = fields_p->A_R[1][idx] = fields_p->A_R[2][idx] = z_H(0.0, 0.0); // 默认液相
            z_H x_vec = z_H(j * params.dx, i * params.dx); // 网格坐标

            // 判断是否在晶粒内部 半径内激活初始波形 
            for (const Seed& s : seeds) {
                double dx_s = (j - s.x) * params.dx;// 到晶粒中心距离
                double dy_s = (i - s.y) * params.dx;
                double distance = sqrt(dx_s * dx_s + dy_s * dy_s);
                
                if (distance < s.radius) {
                    double cos_theta = cos(s.theta);
                    double sin_theta = sin(s.theta);
                    z_H k1_rot = z_H(k1.real() * cos_theta - k1.imag() * sin_theta,
                                     k1.real() * sin_theta + k1.imag() * cos_theta);
                    z_H k2_rot = z_H(k2.real() * cos_theta - k2.imag() * sin_theta,
                                     k2.real() * sin_theta + k2.imag() * cos_theta);
                    z_H k3_rot = z_H(k3.real() * cos_theta - k3.imag() * sin_theta,
                                     k3.real() * sin_theta + k3.imag() * cos_theta);

                    fields_p->A_R[0][idx] = params.A0 * exp( z_H(0, dot_H(k1_rot - k1, x_vec)) );
                    fields_p->A_R[1][idx] = params.A0 * exp( z_H(0, dot_H(k2_rot - k2, x_vec)) );
                    fields_p->A_R[2][idx] = params.A0 * exp( z_H(0, dot_H(k3_rot - k3, x_vec)) );
                }
            }
        }
    }
}

// cpu 重构psi 输出vtk
void reconstruct_output(int step, const Input_Para& params, const PreCal_Para& pre, Fields* fields_p) {
    vector<double> psi_A(pre.Size);
    for (int k = 0; k < pre.Size; ++k) {
        int i = k / params.N;
        int j = k % params.N;
        z_H x_vec = z_H(j * params.dx, i * params.dx);
        z_H A_sum = fields_p->A_R[0][k] * exp( z_H(0, dot_H(pre.k_vec[0], x_vec)) ) +
                    fields_p->A_R[1][k] * exp( z_H(0, dot_H(pre.k_vec[1], x_vec)) ) +
                    fields_p->A_R[2][k] * exp( z_H(0, dot_H(pre.k_vec[2], x_vec)) );
        psi_A[k] = params.psi_c + 2. * A_sum.real();
    }
    // 只有每freq步会把线性项和非线性项合并并转化为psi 并输出
    output_vtk(psi_A, params.N, "output_" + to_string(step) + ".vtk");
}


int main(){
    // 初始化
    Input_Para params;
    PreCal_Para* pre_p;
    Fields* fields_p;
    readInput("input.txt", params);
    params.print();
    initialize(params, pre_p, fields_p);

    // 设置gpu启动参数
    dim3 dimBlock3D = dim3(16, 16, 1);// 线程块：16x16x1个线程
    dim3 dimGrid3D = dim3(16, 16, 1);// 网格：16x16x1个线程块

    // 初始数据传输到gpu
    fields_p->cpMemHtoD(*pre_p);
    
    // gpu 开始模拟
    for (int t = 0; t < params.steps + 1; ++t) {
        /*------------------ 1.gpu计算非线性项 ----------------------*/
        compute_nl_kernel<<< dimGrid3D, dimBlock3D >>>
            (fields_p->d_A_R[0], fields_p->d_A_R[1], fields_p->d_A_R[2], 
             fields_p->d_nl_R[0], fields_p->d_nl_R[1], fields_p->d_nl_R[2], pre_p->Size, pre_p->psi_c_6);
        
        /*------------------ 2.k空间更新 ----------------------*/
        for (int j = 0; j < 3; ++j) {
            // cufft
            cufftExecZ2Z(fields_p->plan_forward, (cufft_z*)fields_p->d_A_R[j], fields_p->d_A_k[j], CUFFT_FORWARD);
            cufftExecZ2Z(fields_p->plan_forward, (cufft_z*)fields_p->d_nl_R[j], fields_p->d_nl_k[j], CUFFT_FORWARD);
            
            // 频域更新
            z_D k_j = make_cuDoubleComplex(pre_p->k_vec[j].real(), pre_p->k_vec[j].imag());
            update_k_kernel<<< dimGrid3D, dimBlock3D >>>
                (fields_p->d_A_k[j], fields_p->d_nl_k[j], k_j, pre_p->Size, params.N, params.dt, pre_p->factor_k, pre_p->Gamma);
        }
        
        /*------------------ 3.icufft输出 归一化 输出 ---------------------*/
        if (t % params.freq == 0 || t == 0) {
            // 只在输出时进行icufft
            for (int j = 0; j < 3; ++j) {
                cufftExecZ2Z(fields_p->plan_backward, fields_p->d_A_k[j], (cufft_z*)fields_p->d_A_R[j], CUFFT_INVERSE); 
            }

            // gpu 归一化
            normalize_kernel<<< dimGrid3D, dimBlock3D >>>(fields_p->d_A_R[0], fields_p->d_A_R[1], fields_p->d_A_R[2], pre_p->Size);
            
            // 传输数据到cpu并输出
            fields_p->cpMemDtoH(*pre_p);
            reconstruct_output(t, params, *pre_p, fields_p);
        }
    } 
    delete fields_p; delete pre_p;
    
    cout << "模拟完成！" << endl;
    return 0;
}