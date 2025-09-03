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
#include "../../FFT/timer.h" // timer类计时

//#include "../FFT/fft_v3_omp.h"
#include <fftw3.h> // 使用fftw库

using namespace std;
namespace fs = std::filesystem;
using z = complex<double>;

const int N = 64; // 需要是2的幂
const int Size = N * N;
const int N_k = N; // 使用完整谱 (c2c)
const double dx = M_PI / 2.0; // 空间步长
const double dt = 0.04; // 时间步长
const int steps = 10000; // 总步数
const int frequency = 1000; // 输出频率

const double psi_c = 0.285; // 平均密度，Fig.1
const double r = -0.25; // 参数r
const double Gamma = -(r + 3 * psi_c * psi_c); // 扰动增长率计算的常数计算Γ
const double k0 = 1.0; // 倒格矢模长，已归一化

// 倒格矢 (k0=1)，参考代码1
const z k1 = { -sqrt(3.0) / 2.0, -0.5 };// 使用复数 用来储存二维数组
const z k2 = { 0.0, 1.0 };
const z k3 = { sqrt(3.0) / 2.0, -0.5 };

// fftw Plan
using fftw_z = fftw_complex;
fftw_plan plan_forward, plan_backward;
fftw_z *in_z, *out_z;
fftw_z *A1_k, *A2_k, *A3_k; // 预分配的频域数组
fftw_z *nl1_k, *nl2_k, *nl3_k;

// 实空间场
vector<z> A1_R(Size), A2_R(Size), A3_R(Size); // 复幅度场
vector<z> nl1_R(Size), nl2_R(Size), nl3_R(Size); // 非线性项


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

/*-------------------------- Seeds 初始化 -------------------------*/
struct Seed {
    int x, y; // 中心坐标
    double theta; // 取向角(弧度)
    double radius; // 半径
};

// 3个不同取向seed，参考代码1
vector<Seed> seeds = { 
    {N/4, N/4, 0, 15},
    {3*N/4, N/4, M_PI/6, 15},
    {N/2, 3*N/4, M_PI/3, 15} 
};

// 初始化seeds 
// (设置A_j)，v4接受的是psi seeds N dx psi_c A_h 这里接收的是三个振幅
void seeds_initialize(vector<z>& A1, vector<z>& A2, vector<z> &A3, const vector<Seed>& seeds, int N, double dx) {
    const double A0 = 0.1; // 初始振幅，参考代码1
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            A1[idx] = A2[idx] = A3[idx] = {0.0, 0.0}; // 默认液相

            // 计算当前网格点的空间坐标向量
            z x_vec = {j * dx, i * dx};// 复数表示二维坐标向量 (x, y)

            // 判断是否在seed范围内（仅半径内激活）
            for (const Seed& s : seeds) {
                double dx_s = (j - s.x) * dx;
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

                    A1[idx] = A0 * exp(z(0, dot(k1_rot - k1, x_vec)));
                    A2[idx] = A0 * exp(z(0, dot(k2_rot - k2, x_vec)));
                    //A3[idx] = A0 * exp(z(0, dot(k3_rot - k3, x_vec)));
                }
            }
        }
    }
}

/*---------------------------------------- fftw ---------------------------------------*/
// fftw 初始化
void fftw_initialize() {
    in_z = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    out_z = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    A1_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    A2_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    A3_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    nl1_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    nl2_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);
    nl3_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * Size);

    plan_forward = fftw_plan_dft_2d(N, N, in_z, out_z, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_backward = fftw_plan_dft_2d(N, N, in_z, out_z, FFTW_BACKWARD, FFTW_ESTIMATE);
}

// fft 
void fft_2d(const vector<z>& input, fftw_z* output) {
    for (int i = 0; i < Size; ++i) {
        in_z[i][0] = input[i].real();
        in_z[i][1] = input[i].imag();
    }
    fftw_execute_dft(plan_forward, in_z, output);
}

// ifft ZtoZ
vector<z> ifft_2d(fftw_z* input) {
    fftw_execute_dft(plan_backward, input, out_z);
    vector<z> output(Size);
    for (int i = 0; i < Size; ++i) {
        output[i] = z(out_z[i][0], out_z[i][1]) / static_cast<double>(Size); // 归一化
    }
    return output;
}

// 频域更新
void update_k(fftw_z* Aj_k, const fftw_z* nl_k, const z& k_j) {
    const double factor_k = 2.0 * M_PI / (N * dx);// 基本波矢单位，将离散格点映射到连续波矢空间
    for (int i = 0; i < N; ++i) {
        double ky = (i < N / 2) ? i * factor_k : (i - N) * factor_k;// 正负频率
        for (int j = 0; j < N; ++j) {
            double kx = (j < N / 2) ? j * factor_k : (j - N) * factor_k;// 正负频率
            int idx = i * N + j;

            z q_0_k = { kx, ky };// 当前波矢，用复数储存二维向量！！
            z q_1_k = q_0_k + k_j;// 经过晶粒波矢k_j移位后的波矢
            double q2 = q_0_k.real() * q_0_k.real() + q_0_k.imag() * q_0_k.imag();// 波矢模长平方
            double k_dot_q = k_j.real() * q_0_k.real() + k_j.imag() * q_0_k.imag();// 波矢与kj点积
            double op_2 = -(q2 + 2 * k_dot_q);// 算子平方项

            // 旋转协变 线性算子Lj
            double Lj_k = (q_1_k.real() * q_1_k.real() + q_1_k.imag() * q_1_k.imag()) * (Gamma - op_2 * op_2);

            // 时间步进 半隐式欧拉法
            z k_Aj (Aj_k[idx][0], Aj_k[idx][1]);
            z k_nl (nl_k[idx][0], nl_k[idx][1]);// 非线性项数据为 main中显式处理
            z Aj_k_new = (k_Aj + dt * k_nl) / (1.0 - dt * Lj_k);// 隐式处理
            
            Aj_k[idx][0] = Aj_k_new.real();
            Aj_k[idx][1] = Aj_k_new.imag();
        }
    }
}

// fftw 清理
void fftw_clean() {
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(in_z); fftw_free(out_z);
    fftw_free(A1_k); fftw_free(A2_k); fftw_free(A3_k);
    fftw_free(nl1_k); fftw_free(nl2_k); fftw_free(nl3_k);
    fftw_cleanup();
}

// 重构psi 输出vtk
void reconstruct_output_psi(int steps) {
    vector<double> psi_reconstructed(Size);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            z x_vec = { j * dx, i * dx };// 复数存储 当前网格点空间物理位置向量坐标

            // 计算相位因子 e^(ik·x) 中的指数部分
            // 计算 e^(ik·x) = cos(k·x) + i*sin(k·x) 这里Aj_R已经初始化好
            z sum_A_exp = A1_R[idx] * exp(z(0, dot(k1, x_vec))) +
                          A2_R[idx] * exp(z(0, dot(k2, x_vec))) +
                          A3_R[idx] * exp(z(0, dot(k3, x_vec)));
            psi_reconstructed[idx] = psi_c + 2.0 * sum_A_exp.real();
        }
    }
    output_vtk(psi_reconstructed, N, "output_" + to_string(steps) + ".vtk");
}

int main() {

    // 初始化
    fftw_initialize();
    Timer timer(steps, frequency);
    seeds_initialize(A1_R, A2_R, A3_R, seeds, N, dx);
    
    vector<z>* A_R[3] = {&A1_R, &A2_R, &A3_R};
    vector<z>* nl_R[3] = {&nl1_R, &nl2_R, &nl3_R};
    fftw_z* A_k[3] = {A1_k, A2_k, A3_k};
    fftw_z* nl_k[3] = {nl1_k, nl2_k, nl3_k};
    const z k_vals[3] = {k1, k2, k3};

    timer.start();
    reconstruct_output_psi(0);

    // 模拟
    for (int t = 0; t < steps + 1; ++t) {
        /*------------------------- 1.计算非线性项 N_j ------------------------------*/
        for (int i = 0; i < Size; ++i) {
            double A_abs2[3] = { norm(A1_R[i]), norm(A2_R[i]), norm(A3_R[i]) };
            z A_conj[3] = { conj(A1_R[i]), conj(A2_R[i]), conj(A3_R[i]) };
            
            nl1_R[i] = -3.0 * A1_R[i] * (A_abs2[0] + 2.0 * (A_abs2[1] + A_abs2[2])) - 
                        6.0 * psi_c * A_conj[1] * A_conj[2];
            nl2_R[i] = -3.0 * A2_R[i] * (A_abs2[1] + 2.0 * (A_abs2[0] + A_abs2[2])) - 
                        6.0 * psi_c * A_conj[0] * A_conj[2];
            nl3_R[i] = -3.0 * A3_R[i] * (A_abs2[2] + 2.0 * (A_abs2[0] + A_abs2[1])) - 
                        6.0 * psi_c * A_conj[0] * A_conj[1];
        }

        /*--------------------------- 2.fft和ifft k空间更新 ------------------------------*/
        for (int j = 0; j < 3; ++j) {
             fft_2d(*A_R[j], A_k[j]);
             fft_2d(*nl_R[j], nl_k[j]);
             update_k(A_k[j], nl_k[j], k_vals[j]);
             *A_R[j] = ifft_2d(A_k[j]);
         }

        /*------------------ 4.根据frequency 重构psi 输出vtk ----------------------*/
        if (t % frequency == 0) {
            timer.update(t);
            reconstruct_output_psi(t);
        }
    }

    timer.finish();
    cout << "模拟完成！" << endl;
    fftw_clean();
    
    return 0;
}