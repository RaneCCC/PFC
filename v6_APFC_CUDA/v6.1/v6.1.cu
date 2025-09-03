/*
2025.8 v6.1 PFC-RG晶粒生长 CUDA版 初始框架
(基于2006 Renormalization Group方法 DOI: 10.1007/s10955-005-9013-7)

相较于v5.3：
1. cuda：
   - cuda复数类型 cuDoubleComplex 替代 complex<double>
   - cuFFT库替代fftw
   - 内存分配以及管理 cudaMalloc cudaMemcpy
   
nvcc -O3 -o apfc_cuda v6.1.cu -lcufft
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

__device__ inline double dot_D(z_D k, z_D x){
   //cuCreal() : 提取复数实部 cuCimag() : 提取复数虚部
    return cuCreal(k) * cuCreal(x) + cuCimag(k) * cuCimag(x);
}

inline double dot_H(z_H k, z_H x){
    return k.real() * x.real() + k.imag() * x.imag();
}

__device__ inline int idx_gpu(int i, int j, int N ){
    return i + N * j;
}


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

    // constructor
    Input_Para(): N(256), dx(M_PI/2.), dt(0.04), steps(10000), freq(1000), 
                  psi_c(0.285), r(-0.25), k0(1.){}

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
    z_H *h_A_R[3], *h_nl_R[3];// cpu 复数域指针
    z_D *d_A_R[3], *d_nl_R[3];// gpu 复数域数据指针
    cufft_z *d_A_k[3], *d_nl_k[3];// gpu 频域数据指针
    cufftHandle plan_forward, plan_backward;

    // constructor 初始化
    Fields(const Input_Para& params, const PreCal_Para& pre)
    {   
        // 初始化cpu数据（也可以在{}外单独设置vector<z_H>）
        for(int i = 0; i < 3; ++i){
            A_R[i].resize(pre.Size);
            nl_R[i].resize(pre.Size);
        }
        
        // 初始化cpu指针
        for(int i = 0; i < 3; ++i){
            h_A_R[i] = reinterpret_cast<z_D*>(A_R[i].data());// 强制转换为CUDA复数指针
            h_nl_R[i] = reinterpret_cast<z_D*>(nl_R[i].data());

            cudaMalloc(&d_A_R[i], sizeof(z_D) * pre.Size);
            cudaMalloc(&d_nl_R[i], sizeof(z_D) * pre.Size);
            cudaMalloc(&d_A_k[i], sizeof(z_D) * pre.Size);
            cudaMalloc(&d_nl_k[i], sizeof(z_D) * pre.Size);
        }

        // 初始化cufft Plan 
        // (计划指针, 宽度, 高度, 变换类型) 双精度复数到复数的变换（Z = double complex）
        cufftPlan2d(&plan_forward, params.N, params.N, CUFFT_Z2Z);
        cufftPlan2d(&plan_backward, params.N, params.N, CUFFT_Z2Z);
    }

    // 数据传输 HtoD
    void cpMemHtoD(const PreCal_Para& pre){
        for(int i = 0; i < 3; ++i){
            cudaMemcpy(d_A_R[i], h_A_R[i], sizeof(z_D) * pre.Size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_nl_R[i], h_nl_R[i], sizeof(z_D) * pre.Size, cudaMemcpyHostToDevice);
        }
    }

    // 数据传输 DtoH
    void cpMemDtoH(const PreCal_Para& pre){
        for(int i = 0; i < 3; ++i){
            cudaMemcpy(h_A_R[i], d_A_R[i], sizeof(z_D) * pre.Size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_nl_R[i], d_nl_R[i], sizeof(z_D) * pre.Size, cudaMemcpyDeviceToHost);
        }
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

// 全局参数对象
Input_Para params;
PreCal_Para* pre_p = nullptr;
Fields* fields = nullptr;



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

    infile.close();
    cout << "Input read from " << filename << endl;
}

// 创建输出文件夹 返回输出路径
string get_output_path(const string& filename) {
    string folder = "output"; // 创建output文件夹
    if (!std::filesystem::exists(folder)) { std::filesystem::create_directory(folder); }
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


/*-------------------------- 主要函数 -----------------------------*/
// 频域更新
__global__ void update_k_kernel(cufft_z* Aj, const cufft_z* nl, z_D k_j, int N, double dt, double factor_k, double Gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N) return;

    double ky = (i < N / 2) ? i * factor_k : (i - N) * factor_k;
    double kx = (j < N / 2) ? j * factor_k : (j - N) * factor_k;
    int idx = idx_gpu(i, j, N);

    z_D q0 = make_cuDoubleComplex(kx, ky);// 将C++的double值转换为CUDA复数类型
    z_D q1 = cuCadd(q0, k_j);// 经过晶粒波矢k_j移位后的波矢
    double op_2 = -(dot_D(q0, q0) + 2 * dot_D(k_j, q0));// 算子平方项

    // 旋转协变 线性算子Lj
    double Lj = (dot_D(q1, q1)) * (Gamma - op_2 * op_2);

    // 时间步进 半隐式欧拉法
    z_D Aj_k = Aj[idx];
    z_D nl_k = nl[idx];
    z_D dt_nl = cuCmul(make_cuDoubleComplex(dt, 0), nl_k);// dt*nl
    z_D numerator = cuCadd(Aj_k, dt_nl);// Aj + dt*nl_k
    double denominator = 1.0 - dt * Lj;
    z_D Aj_new = cuCdiv(numerator, make_cuDoubleComplex(denominator, 0));//(Aj + dt * nl_k) / (1. - dt * Lj)
    
    Aj[idx] = Aj_new;
}

void update_k(cufft_z* d_A_k, cufft_z* d_nl_k, z_D k_j, const Input_Para& params, const PreCal_Para& pre){
    dim3 blockSize(16, 16);
    dim3 gridSize((params.N + blockSize.x - 1) / blockSize.x,
                  (params.N + blockSize.y - 1) / blockSize.y);
    
    update_k_kernel<<<gridSize, blockSize>>>(d_A_k, d_nl_k, k_j, params.N, params.dt, pre.factor_k, pre.Gamma);
    cudaDeviceSynchronize();
}



// 初始化
void initialize(const Input_Para& params) {
    // 初始化1 预计算参数 fft 场变量
    pre_p = new PreCal_Para(params);
    fields_p = new Fields(params, *pre_p);

    // 初始化2 倒格矢
    z_H k1 = z_H(-sqrt(3.0) / 2.0, -0.5);
    z_H k2 = z_H(0.0, 1.0);
    z_H k3 = z_H(sqrt(3.0) / 2.0, -0.5);
    pre_p->k_vec = { k1, k2, k3 };
    
    // 初始化3 种子
    struct Seed { int x, y; double theta; double radius; };
    vector<Seed> seeds = {
        {   params.N/4,     params.N/4,      0,     15 },
        { 3*params.N/4,     params.N/4,    M_PI/6,  15 },
        {   params.N/2,   3*params.N/4,    M_PI/3,  15 }};
    const double A0 = 0.1; // 初始振幅
    
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

                    fields_p->A_R[0][idx] = A0 * exp(z_H(0, dot_H(k1_rot - k1, x_vec)));
                    fields_p->A_R[1][idx] = A0 * exp(z_H(0, dot_H(k2_rot - k2, x_vec)));
                    fields_p->A_R[2][idx] = A0 * exp(z_H(0, dot_H(k3_rot - k3, x_vec)));
                }
            }
        }
    }
}


// 重构psi 输出vtk
void reconstruct_output(int step, const Input_Para& params, const PreCal_Para& pre) {
    vector<double> psi_A(pre.Size);
    for (int k = 0; k < pre.Size; ++k) {
        // 实时计算相位因子
        int i = k / params.N;
        int j = k % params.N;
        z_H x_vec = z_H(j * params.dx, i * params.dx);
        z_H A_sum = fields_p->A_R[0][k] * exp(z_H(0, dot_H(pre.k_vec[0], x_vec))) +
                    fields_p->A_R[1][k] * exp(z_H(0, dot_H(pre.k_vec[1], x_vec))) +
                    fields_p->A_R[2][k] * exp(z_H(0, dot_H(pre.k_vec[2], x_vec)));
        psi_A[k] = params.psi_c + 2.0 * A_sum.real();
    }
    // 只有每freq步会把线性项和非线性项合并并转化为psi 并输出
    output_vtk(psi_A, params.N, "output_" + to_string(step) + ".vtk");
}


int main(){
    Input_Para params;
    readInput("config.txt", params);
    params.print();
    initialize(params);

    for (int t = 0; t < params.steps + 1; ++t) {
        /*------------------ 1.实空间计算非线性项 N_j ------------------------------*/
        for (int i = 0; i < pre_p->Size; ++i) {
            double A_abs2[3] = { norm(fields_p->A_R[0][i]), 
                                 norm(fields_p->A_R[1][i]), 
                                 norm(fields_p->A_R[2][i]) };
            z_H A_conj[3] = { conj(fields_p->A_R[0][i]), 
                              conj(fields_p->A_R[1][i]), 
                              conj(fields_p->A_R[2][i]) };
            
            fields_p->nl_R[0][i] = -3.0 * fields_p->A_R[0][i] * (A_abs2[0] + 2.0 * (A_abs2[1] + A_abs2[2])) - 
                                 pre_p->psi_c_6 * A_conj[1] * A_conj[2];
            fields_p->nl_R[1][i] = -3.0 * fields_p->A_R[1][i] * (A_abs2[1] + 2.0 * (A_abs2[0] + A_abs2[2])) - 
                                 pre_p->psi_c_6 * A_conj[0] * A_conj[2];
            fields_p->nl_R[2][i] = -3.0 * fields_p->A_R[2][i] * (A_abs2[2] + 2.0 * (A_abs2[0] + A_abs2[1])) - 
                                 pre_p->psi_c_6 * A_conj[0] * A_conj[1];
        }

        /*----------------------- 2.传输数据到gpu --------------------------*/
        fields_p->cpMemHtoD(*pre_p);
        
        /*--------------------------- 2.k空间更新 ------------------------------*/
        for (int j = 0; j < 3; ++j) {
            // cufft
            cufftExecZ2Z(fields_p->plan_forward, (cufft_z*)fields_p->d_A_R[j], fields_p->d_A_k[j], CUFFT_FORWARD);
            cufftExecZ2Z(fields_p->plan_forward, (cufft_z*)fields_p->d_nl_R[j], fields_p->d_nl_k[j], CUFFT_FORWARD);
            
            // 频域更新
            z_D k_j = make_cuDoubleComplex(pre_p->k_vec[j].real(), pre_p->k_vec[j].imag());
            update_k(fields_p->d_A_k[j], fields_p->d_nl_k[j], k_j, params, *pre_p);
            
            // icufft
            cufftExecZ2Z(fields_p->plan_backward, fields_p->d_A_k[j], (cufft_z*)fields_p->d_A_R[j], CUFFT_INVERSE);
        }
        
        /*----------------------- 4.传输数据到cpu --------------------------*/
        fields->cpMemDtoH(*pre_p);
        
        /*----------------------- 5.归一化 icufft --------------------------*/
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < pre_p->Size; ++i) {
                fields_p->A_R[j][i] /= static_cast<double>(pre_p->Size);
            }
        }

        /*------------------ 6.根据freq 重构psi 输出vtk ----------------------*/
        if (t % params.freq == 0 || t == 0) {
            reconstruct_output(t, params, *pre_p);
        }
    }

    cout << "模拟完成！" << endl;
    
    // 清理内存
    delete fields;
    delete pre_p;
    
    return 0;
}