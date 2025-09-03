/*
2025.8 PFC晶粒生长
使用fftw库
clang++ -o pfc main.cpp -std=c++17 -I../FFT/include -L../FFT/lib -lfftw3
clang++ -O3 -o pfc main.cpp -std=c++17 -I../FFT/include -L../FFT/lib -lfftw3

*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream> 
#include <complex> 
#include <filesystem>// 创建输出文件夹
#include "../FFT/timer.h"// timer类计时

//#include "../FFT/fft_v1_simple.h"
//#include "../FFT/fft_v2_iteration.h"
//#include "../FFT/fft_v3_omp.h"
#include <fftw3.h>// 使用fftw库

using namespace std;
namespace fs = std::filesystem;//新名称空间用来在上级目录新建文件夹储存数据

const int N = 128;// 需要是2的幂 不然递归调用整除不完 256
const int Size = N * N;
const int N_k = N / 2 + 1;
const double dx = M_PI / 4.0;// 空间步长
const double dt = 0.01;// 减小时间步长保证稳定性（原0.01）
const int steps = 15001;
const double epsilon = 4.0 / 15.0;// 控制参数

const double psi_c = 0.2;// 均匀相背景值
const double A_s = -0.3;// 条纹相振幅
const double A_h = -4.0 / 5.0 * sqrt((15 * epsilon - 36 * psi_c * psi_c)/ 3.);// 六方相振幅 -0.326
const double q0 = 1.0;// 特征波数 2pi/T
const double q_s = 2.0 * sqrt(epsilon - 3 * psi_c * psi_c);// 条纹相 0.562
const double q_h = sqrt(3.) * q0 / 2.;// 六方相 0.8660
const int frequency = 0;// 输出频率

// fftw Plan
using fftw_z = fftw_complex;
fftw_plan plan_forward, plan_backward;
double *in_R, *out_R;
fftw_z *out_Z, *in_Z;
fftw_z *psi_k, *nl_k;// 预分配的频域数组


// 创建输出文件夹 返回输出路径
string get_output_path(const string& filename) {
    string folder = "output";// 创建outdata文件夹，确保存在，返回路径
    if (!fs::exists(folder))  { fs::create_directory(folder); }
    return folder + "/" + filename;
}

void output_vtk(const vector<double>& psi, int N, const string& filename) {
    string full_path = get_output_path(filename);
    ofstream fout_vtk(full_path);
    
    fout_vtk << "# vtk DataFile Version 3.0\n";
    fout_vtk << "PFC simulation results\n";
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


/*--------------------- Seeds 初始化 --------------------*/
struct Seed {
    int x, y;// 中心坐标
    double theta;// 取向角(弧度)
    double radius;// 半径
};

// 3个不同取向seed
vector<Seed> seeds = { {N/4, N/4, 0, 10}, 
                       {3*N/4, N/4, M_PI/4, 10}, 
                       {N/2, 3*N/4, 1*M_PI/2, 10} };

// 生成带取向的六方相密度分布（仅在半径内激活）
double hex_phase(double x, double y, double theta, double A) {
    double xr = x * cos(theta) - y * sin(theta);
    double yr = x * sin(theta) + y * cos(theta);
    return A * ( (cos(q_h * xr) * cos(q_h * yr / sqrt(3.)) + 0.5 * cos(2 * q_h * yr / sqrt(3.))) );// 含取向
    //return A * ( (cos(q_h * x) * cos(q_h * y / sqrt(3.)) + 0.5 * cos(2 * q_h * y / sqrt(3.))) );// 无取向
}

// 初始化seeds
void seeds_initialize(vector<double>& psi, const vector<Seed>& seeds,int N, double dx, double psi_c, double A_h) {
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < N; ++j) { 
            double x = (double)j * dx;
            double y = (double)i * dx;
            int idx = i * N + j;
            double val = psi_c;// 初始液相，无扰动
                                   
            // 判断是否在seed范围内（仅半径内激活六方相）
            for (const Seed& s : seeds) {
                double dx_seed = (j - s.x) * dx;
                double dy_seed = (i - s.y) * dx;
                double dist = sqrt(dx_seed*dx_seed + dy_seed*dy_seed);
                
                //仅半径内添加六方相，移除外部5扰动
                if(dist < s.radius) val += hex_phase(dx_seed, dy_seed, s.theta, A_h);
            }

            psi[idx] = val;
        }
    }
}


/*---------------------------------------- fftw ---------------------------------------*/
// fftw 初始化：（预）分配内存 Plan
void fftw_initialize() {
    // 分配内存
    in_R = (double*) fftw_malloc(sizeof(double) * Size);
    out_Z = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N_k);
    in_Z = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N_k);
    out_R = (double*) fftw_malloc(sizeof(double) * Size);
    
    psi_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * N * N_k);// 预分配频域数组
    nl_k = (fftw_z*) fftw_malloc(sizeof(fftw_z) * N * N_k);
    
    // 创建fftw Plan
    plan_forward = fftw_plan_dft_r2c_2d(N, N, in_R, out_Z, FFTW_ESTIMATE);
    plan_backward = fftw_plan_dft_c2r_2d(N, N, in_Z, out_R, FFTW_ESTIMATE);
}

// fft RtoZ
void fft_RtoZ(const vector<double>& input, fftw_z* result) {
    for (int i = 0; i < Size; ++i) { in_R[i] = input[i]; }
    fftw_execute(plan_forward);
    for (int i = 0; i < N * N_k; ++i) {
        result[i][0] = out_Z[i][0];
        result[i][1] = out_Z[i][1];
    }
}

// fftw 清理
void fftw_clean() {
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    
    fftw_free(in_R);fftw_free(out_Z);
    fftw_free(in_Z);fftw_free(out_R);
    fftw_free(psi_k);fftw_free(nl_k);// 释放预分配的频域数组
    fftw_cleanup();
}

// 频域更新
void update_k(fftw_z* psi_k, const fftw_z* nl_k) {
    const double k_factor = 2.0 * M_PI / (N * dx);
    const double q0_2 = q0 * q0;
    
    for (int i = 0; i < N; ++i) {
        const double ky = ((i <= N / 2) ? i : (i - N)) * k_factor;
        
        for (int j = 0; j < N_k; ++j) {
            const double kx = j * k_factor;
            const double k2 = kx * kx + ky * ky;
            const double factor1 = q0_2 - k2;
            const double Lk = -k2 * (factor1 * factor1 - epsilon);
            const double factor2 = 1.0 - dt * Lk;
            
            const int idx = i * N_k + j;
            const double dt_k2 = dt * k2;
            
            psi_k[idx][0] = (psi_k[idx][0] - dt_k2 * nl_k[idx][0]) / factor2;
            psi_k[idx][1] = (psi_k[idx][1] - dt_k2 * nl_k[idx][1]) / factor2;
            //psi_k[idx][0] = psi_k[idx][0] / factor2;
            //psi_k[idx][1] = psi_k[idx][1] / factor2;
        }
    }
}



int main() {
    vector<double> psi(Size);// 实空间 psi
    vector<double> nl_R(Size);// 实空间 非线性项(nonlinear) psi^3
    
    // fftw 初始化
    fftw_initialize();

    // Timer 初始化
    Timer timer(steps, frequency);
    timer.start();

    // seeds 初始化
    seeds_initialize(psi, seeds, N, dx, psi_c, A_h); 

    // 模拟
    // psi_k[行][列] -> psi_k[索引][R/I] || 二维复数数组 -> 一维实数对数组
    for (int t = 0; t < steps; ++t) {
        /*------------------------- 1.填充nl: psi^3 ------------------------------*/
        for (int i = 0; i < Size; ++i) { nl_R[i] = psi[i] * psi[i] * psi[i]; }

        /*--------------------------- 2.2d fft ------------------------------*/
        fft_RtoZ(psi, psi_k);
        fft_RtoZ(nl_R, nl_k);

        /*-------------------- 3.Fourier空间 更新psi_k（频域） -----------------------*/
        update_k(psi_k, nl_k);

        /*---------------- 4.ifft psi_k -> psi ----------------*/
        for (int i = 0; i < N * N_k; ++i) {
            in_Z[i][0] = psi_k[i][0];
            in_Z[i][1] = psi_k[i][1];
        }
        fftw_execute(plan_backward);
        
        /*------------------- 5.归一化 ------------------*/
        for (int i = 0; i < Size; ++i) { psi[i] = out_R[i] / Size; } // FFTW不自动归一化
           
        /*------------------ 6.输出vtk ----------------------*/
        if (t % frequency == 0) {
            timer.update(t);
            output_vtk(psi, N, "output_" + to_string(t) + ".vtk");
        }
    }

    timer.finish();
    cout << "模拟完成！" << endl;
    fftw_clean();
    
    return 0;
}