/*
2025.8 PFC-RG晶粒生长(基于2006 Renormalization Group方法 DOI: 10.1007/s10955-005-9013-7) 
演化复幅度A_j，重建密度场ψ
clang++ -O3 -o apfc main.cpp -std=c++17 -I../../FFT/include -L../../FFT/lib -lfftw3
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream> 
#include <complex> 
#include <filesystem>
#include "timer.h" // timer类计时

//#include "../FFT/fft_v3_omp.h"
#include <fftw3.h> // 使用fftw库

using namespace std;
namespace fs = std::filesystem;
using z = complex<double>;
using fftw_z = fftw_complex;
const int size_fftw_z = sizeof(fftw_complex);

const int N = 64; // 需要是2的幂
const int Size = N * N;
const double dx = M_PI / 2.0; // 空间步长
const double dt = 0.04; // 时间步长
const int steps = 10000; // 总步数
const int frequency = 1000; // 输出频率

const double psi_c = 0.285; // 平均密度，Fig.1
const double r = -0.25; // 参数r
const double Gamma = -(r + 3 * psi_c * psi_c); // 扰动增长率计算的常数计算Γ
const double factor_k = 2.0 * M_PI / (N * dx);// 基本波矢单位，将离散格点映射到连续波矢空间
const double k0 = 1.0; // 倒格矢模长，已归一化
const double A0 = 0.1; // 初始振幅

// 倒格矢 (k0=1)，参考代码1
const z k1 = { -sqrt(3.0) / 2.0, -0.5 };// 使用复数 用来储存二维数组
const z k2 = { 0.0, 1.0 };
const z k3 = { sqrt(3.0) / 2.0, -0.5 };
const vector<z> k_vec = { k1, k2, k3 };


// apfc 场变量结构类(包含fftw内存和计划的创建)
class Fields{
public:
    // fftw 空间中的内存分配
    fftw_plan plan_forward, plan_backward;
    fftw_z *in_z, *out_z;
    fftw_z *A_k[3];
    fftw_z *nl_k[3];
    
    // 实空间的振幅和非线性项
    vector<z> A_R[3]; 
    vector<z> nl_R[3]; 
    const z k[3] = { k1, k2, k3 };// 倒格矢
    
    // 构造函数 初始化
    Fields() : 
        A_R{ vector<z>(Size), vector<z>(Size), vector<z>(Size) },
        nl_R{ vector<z>(Size), vector<z>(Size), vector<z>(Size) },
        k{ k1, k2, k3 }
    {}
};
Fields v;


// 创建输出文件夹 返回输出路径
string get_output_path(const string& filename) {
    string folder = "output"; // 创建output文件夹
    if (!fs::exists(folder)) { fs::create_directory(folder); }
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

// 定义点积，计算两个复数表示的二维向量的点积
inline double dot(z k, z x) {
    return k.real() * x.real() + k.imag() * x.imag();
}

/*---------------------- Seeds 初始化 -----------------------*/
struct Seed { int x, y; double theta; double radius; };

// 3个不同取向seed，参考v4
vector<Seed> seeds = { 
    {N/4, N/4, 0, 15},
    {3*N/4, N/4, M_PI/6, 15},
    {N/2, 3*N/4, M_PI/3, 15} };

// 初始化seeds 
// (设置A_j)，v4接受的是psi seeds N dx psi_c A_h 这里接收的是三个振幅
void seeds_initialize(vector<z>& A1_R, vector<z>& A2_R, vector<z>& A3_R, // 这里接收A_R[0.1.2]
                      const vector<Seed>& seeds, int N, double dx) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            A1_R[idx] = A2_R[idx] = A3_R[idx] = { 0.0, 0.0 }; // 默认液相

            z x_vec = { j * dx, i * dx };// 计算当前网格点的二维空间坐标 用复数表示

            // 判断是否在晶粒内部 半径内激活初始波形 
            for (const Seed& s : seeds) {
                double dx_s = (j - s.x) * dx;// 到晶粒中心距离
                double dy_s = (i - s.y) * dx;
                double dist = sqrt(dx_s * dx_s + dy_s * dy_s);
                
                if (dist < s.radius) {
                    // 原始PFC的旋转 只需要设置好角度 初始波形添加即可 但是APFC需要乘相位因子e^iΔkx
                    // 旋转倒格矢：R(θ)xk_j= e^-ikj·x =(k_x cosθ - k_y sinθ, k_x sinθ + k_y cosθ)
                    double cos_theta = cos(s.theta);
                    double sin_theta = sin(s.theta);
                    z k1_rot = {k1.real() * cos_theta - k1.imag() * sin_theta,
                                k1.real() * sin_theta + k1.imag() * cos_theta};
                    z k2_rot = {k2.real() * cos_theta - k2.imag() * sin_theta,
                                k2.real() * sin_theta + k2.imag() * cos_theta};
                    z k3_rot = {k3.real() * cos_theta - k3.imag() * sin_theta,
                                k3.real() * sin_theta + k3.imag() * cos_theta};

                    A1_R[idx] = A0 * exp(z(0, dot(k1_rot - k1, x_vec)));
                    A2_R[idx] = A0 * exp(z(0, dot(k2_rot - k2, x_vec)));
                    //A3_R[idx] = A0 * exp(z(0, dot(k3_rot - k3, x_vec)));
                }
            }
        }
    }
}

/*---------------------------------------- fftw ---------------------------------------*/
// fftw 初始化
void fftw_initialize() {
    // 创建fftw内存
    v.in_z = (fftw_z*) fftw_malloc(size_fftw_z * Size);
    v.out_z = (fftw_z*) fftw_malloc(size_fftw_z * Size);
    for(int i = 0; i < 3; ++i)  {
        v.A_k[i] = (fftw_z*) fftw_malloc(size_fftw_z * Size);
        v.nl_k[i] = (fftw_z*) fftw_malloc(size_fftw_z * Size);
    }
        
    // 初始化fft Plan
    v.plan_forward = fftw_plan_dft_2d(N, N, v.in_z, v.out_z, FFTW_FORWARD, FFTW_ESTIMATE);
    v.plan_backward = fftw_plan_dft_2d(N, N, v.in_z, v.out_z, FFTW_BACKWARD, FFTW_ESTIMATE);
}

// fft vector<z> -> fftw_z*
void fft_2d(const vector<z>& input, fftw_z* output) {
    for (int i = 0; i < Size; ++i) {
        v.in_z[i][0] = input[i].real();
        v.in_z[i][1] = input[i].imag();
    }
    fftw_execute_dft(v.plan_forward, v.in_z, output);
}

// ifft fftw_z* -> vector<z>
void ifft_2d(fftw_z* input, vector<z>& output) {
    fftw_execute_dft(v.plan_backward, input, v.out_z);
    for (int i = 0; i < Size; ++i) {
        output[i] = z(v.out_z[i][0], v.out_z[i][1]) / static_cast<double>(Size);
    }
}

// 频域更新
void update_k(fftw_z* Aj, const fftw_z* nl, const z& k_j) {
    for (int i = 0; i < N; ++i) {
        double ky = (i < N / 2) ? i * factor_k : (i - N) * factor_k;// 正负频率
        for (int j = 0; j < N; ++j) {
            double kx = (j < N / 2) ? j * factor_k : (j - N) * factor_k;// 正负频率
            int idx = i * N + j;

            z q0 = { kx, ky };// 当前波矢，用复数储存二维向量！！
            z q1 = q0 + k_j;// 经过晶粒波矢k_j移位后的波矢
            //double q1_2 = dot(q1, q1);
            //double op_2 = (1 - q1_2) * (1 - q1_2);
            double op_2 = -(dot(q0, q0) + 2 * dot(k_j, q0));// 算子平方项

            // 旋转协变 线性算子Lj
            //double Lj = (dot(q1, q1)) * (Gamma - op_2);
            double Lj = (dot(q1, q1)) * (Gamma - op_2 * op_2);

            // 时间步进 半隐式欧拉法
            z Aj_k(Aj[idx][0], Aj[idx][1]);
            z nl_k(nl[idx][0], nl[idx][1]);// 非线性项数据 在main中显式处理 这里只是传递数据
            z Aj_new = (Aj_k + dt * nl_k) / (1. - dt * Lj);// 复数运算 隐式处理
            
            Aj[idx][0] = Aj_new.real();
            Aj[idx][1] = Aj_new.imag();
        }
    }
}

// 重构psi 输出vtk
void reconstruct_output(int steps) {
    vector<double> psi_A(Size);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            z x_vec = { j * dx, i * dx };// 复数存储 当前网格点空间物理位置向量坐标

            z A_sum = v.A_R[0][idx] * exp(z(0, dot(k1, x_vec))) +
                          v.A_R[1][idx] * exp(z(0, dot(k2, x_vec))) +
                          v.A_R[2][idx] * exp(z(0, dot(k3, x_vec)));
            psi_A[idx] = psi_c + 2.0 * A_sum.real();
        }
    }
    // 只有每frequency步会把线性项和非线性项合并并转化为psi 并输出
    output_vtk(psi_A, N, "output_" + to_string(steps) + ".vtk");
}

// fftw 清理
void fftw_clean() {
    fftw_destroy_plan(v.plan_forward);
    fftw_destroy_plan(v.plan_backward);
    fftw_free(v.in_z); fftw_free(v.out_z);
    fftw_free(v.A_k[0]); fftw_free(v.A_k[1]); fftw_free(v.A_k[2]);
    fftw_free(v.nl_k[0]); fftw_free(v.nl_k[1]); fftw_free(v.nl_k[2]);
    fftw_cleanup();
}

int main() {
    // 初始化
    fftw_initialize();
    Timer timer(steps, frequency);
    seeds_initialize(v.A_R[0], v.A_R[1], v.A_R[2], seeds, N, dx);
    timer.start();
    reconstruct_output(0);

    // 模拟
    for (int t = 1; t < steps + 1; ++t) {
        /*------------------------- 1.实空间计算非线性项 N_j ------------------------------*/
        for (int i = 0; i < Size; ++i) {
            double A_abs2[3] = { norm(v.A_R[0][i]), norm(v.A_R[1][i]), norm(v.A_R[2][i]) };// norm算复数模长平方
            z A_conj[3] = { conj(v.A_R[0][i]), conj(v.A_R[1][i]), conj(v.A_R[2][i]) };
            
            v.nl_R[0][i] = -3.0 * v.A_R[0][i] * (A_abs2[0] + 2.0 * (A_abs2[1] + A_abs2[2])) - 
                            6.0 * psi_c * A_conj[1] * A_conj[2];
            v.nl_R[1][i] = -3.0 * v.A_R[1][i] * (A_abs2[1] + 2.0 * (A_abs2[0] + A_abs2[2])) - 
                            6.0 * psi_c * A_conj[0] * A_conj[2];
            v.nl_R[2][i] = -3.0 * v.A_R[2][i] * (A_abs2[2] + 2.0 * (A_abs2[0] + A_abs2[1])) - 
                            6.0 * psi_c * A_conj[0] * A_conj[1];
        }

        /*--------------------------- 2.fft和ifft k空间更新 ------------------------------*/
        for (int j = 0; j < 3; ++j) {
             fft_2d(v.A_R[j], v.A_k[j]);
             fft_2d(v.nl_R[j], v.nl_k[j]);
             update_k(v.A_k[j], v.nl_k[j], v.k[j]);
             ifft_2d(v.A_k[j], v.A_R[j]);
         }

        /*------------------ 3.根据frequency 重构psi 输出vtk ----------------------*/
        if (t == steps || t % frequency == 0) {
            timer.update(t);
            reconstruct_output(t);
        }
    }

    timer.finish();
    cout << "模拟完成！" << endl;
    fftw_clean();
    
    return 0;
}