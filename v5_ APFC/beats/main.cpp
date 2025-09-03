/*
2025.8 PFC晶粒生长 - Beats干涉版本
使用自定义fft_beats.h实现单FFT分量对六方相的干涉处理
clang++ -o pfc main.cpp -std=c++17 -fopenmp
clang++ -O3 -march=native -ffast-math -funroll-loops -o pfc main.cpp -std=c++17 -fopenmp
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream> 
#include <complex> 
#include <filesystem>
#include "../../FFT/timer.h"
#include "../../FFT/fft_beats.h"  // 使用专门的beats FFT

using namespace std;
namespace fs = std::filesystem;

const int N = 128;
const int Size = N * N;
const int N_k = N / 2 + 1;
const double dx = M_PI / 4.0;
const double dt = 0.005;  // 进一步减小时间步长
const int steps = 8001;
const double epsilon = 4.0 / 15.0;

const double psi_c = 0.2;
const double A_s = -0.3;
const double A_h = -4.0 / 5.0 * sqrt((15 * epsilon - 36 * psi_c * psi_c)/ 3.0) * 0.1;  // 进一步减小六方相幅度

const double q0 = 1.0;
const double q_s = 2.0 * sqrt(epsilon - 3 * psi_c * psi_c);
const double q_h = sqrt(3.0) * q0 / 2.0;
const int frequency = 1000;

// 创建输出文件夹
string get_output_path(const string& filename) {
    string folder = "output";
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
    }
    return folder + "/" + filename;
}

// Seed结构体
struct Seed {
    int x, y;
    double theta;
    double radius;
    bool interference;  // 是否产生干涉
};

// 3个不同取向seed - 设置干涉标志
vector<Seed> seeds = { {N/4, N/4, 0, 12, false},           // 无干涉对照组
                       {3*N/4, N/4, M_PI/3, 12, true},     // 60度取向，产生干涉
                       {N/2, 3*N/4, 2*M_PI/3, 12, true} }; // 120度取向，产生干涉

// 生成带取向的六方相密度分布
double hex_phase(double x, double y, double theta, double A) {
    double xr = x * cos(theta) - y * sin(theta);
    double yr = x * sin(theta) + y * cos(theta);
    return A * ( (cos(q_h * xr) * cos(q_h * yr / sqrt(3.)) + 0.5 * cos(2 * q_h * yr / sqrt(3.))) );
}

// 生成干涉掩码
vector<double> generate_interference_mask(int N) {
    vector<double> mask(Size, 0.0);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            
            // 检查是否在干涉晶粒范围内
            for (const Seed& s : seeds) {
                if (!s.interference) continue;
                
                double dx_seed = (j - s.x);
                double dy_seed = (i - s.y);
                double dist = sqrt(dx_seed*dx_seed + dy_seed*dy_seed);
                
                if (dist < s.radius * 0.8) {  // 缩小干涉范围
                    // 根据距离和角度计算干涉强度
                    double angle = atan2(dy_seed, dx_seed);
                    double interference_strength = exp(-dist / (s.radius * 0.5)) * 0.3;  // 减弱干涉强度
                    
                    // 添加方向性干涉模式
                    double directional_factor = 0.5 * (1.0 + cos(3.0 * (angle - s.theta)));
                    mask[idx] = max(mask[idx], interference_strength * directional_factor);
                }
            }
        }
    }
    
    return mask;
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
    
    for (int i = 0; i < N * N; ++i) {fout_vtk << psi[i] << "\n";}

    fout_vtk.close();
    cout << "Output file: " << filename << endl;
}

int main() {
    vector<double> psi(Size);
    vector<double> nl_R(Size);
    Timer timer(steps, frequency);
    
    // 预分配FFT内存
    vector<vector<z>> psi_k(N, vector<z>(N_k));
    vector<vector<z>> nl_k(N, vector<z>(N_k));
    
    // 生成干涉掩码
    vector<double> interference_mask = generate_interference_mask(N);

    // 初始化seeds
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < N; ++j) { 
            double x = (double)j * dx;
            double y = (double)i * dx;
            int idx = i * N + j;
            double val = psi_c;

            // 判断是否在seed范围内
            for (const Seed& s : seeds) {
                double dx_seed = (j - s.x) * dx;
                double dy_seed = (i - s.y) * dx;
                double dist = sqrt(dx_seed*dx_seed + dy_seed*dy_seed);
                
                if(dist < s.radius) {
                    val += hex_phase(dx_seed, dy_seed, s.theta, A_h);
                }
            }

            psi[idx] = val;
        }
    } 

    // 检查初始状态的数值稳定性
    double max_val = 0.0, min_val = 0.0;
    for (int i = 0; i < Size; ++i) {
        max_val = max(max_val, abs(psi[i]));
        if (isnan(psi[i]) || isinf(psi[i])) {
            cout << "Warning: Invalid value detected at initialization, position " << i << endl;
            psi[i] = psi_c;  // 重置为背景值
        }
    }
    cout << "Initial max |psi|: " << max_val << endl;
    
    // 如果初始值过大，进行归一化
    if (max_val > 10.0) {
        cout << "Normalizing initial values..." << endl;
        for (int i = 0; i < Size; ++i) {
            psi[i] = psi[i] / max_val * 2.0;  // 归一化到合理范围
        }
    }

    // 模拟主循环
    timer.start();
    for (int t = 0; t < steps; ++t) {
        /*------------------------- 1.填充nl: psi^3 ------------------------------*/
        for (int i = 0; i < Size; ++i) { nl_R[i] = psi[i] * psi[i] * psi[i]; }

        /*--------------------------- 2.2d fft with beats effect ------------------------------*/
        // 对psi使用带干涉的FFT，对非线性项使用标准FFT
        psi_k = fft_2d_beats(psi, N, interference_mask);
        nl_k = fft_2d(nl_R, N);

        /*-------------------- 3.Fourier空间 更新psi_k（频域） -----------------------*/
        for (int i = 0; i < N; ++i) {
            double ky = (i <= N / 2) ? i : (i - N);
            ky *= 2.0 * M_PI / (N * dx);

            for (int j = 0; j < N_k; ++j) {
                double kx = j * 2.0 * M_PI / (N * dx);
                double k2 = kx * kx + ky * ky;
                double factor1 = q0 * q0 - k2;
                double Lk = -k2 * (factor1 * factor1 - epsilon);

                z factor2 = 1.0 - dt * z(Lk, 0);
                
                // 数值稳定性检查：避免除以接近零的数
                if (abs(factor2) > 1e-12) {
                    psi_k[i][j] = (psi_k[i][j] - dt * z(k2, 0) * nl_k[i][j]) / factor2;
                } else {
                    // 当分母接近零时，使用更稳定的更新方式
                    psi_k[i][j] = psi_k[i][j] * 0.99;  // 轻微衰减
                }
                
                // 限制频域分量的幅度，防止数值爆炸
                if (abs(psi_k[i][j]) > 1e6) {
                    psi_k[i][j] = psi_k[i][j] / abs(psi_k[i][j]) * 1e6;
                }
            } 
        }

        /*---------------- 4.ifft psi_k -> psi ----------------*/
        psi = ifft_2d(psi_k, N);   

        /*------------------ 5.输出vtk ----------------------*/
        timer.update(t);
        if (t % frequency == 0) { 
            output_vtk(psi, N, "output_" + to_string(t) + ".vtk"); 
        }
    }

    timer.finish();
    cout << "Beats干涉模拟完成！" << endl;
    return 0;
}