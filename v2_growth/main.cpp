/*
2025.7 PFC晶粒生长
使用自己写的简化fft.h
clang++ -o pfc main.cpp -std=c++17
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream> 
#include <complex> 
#include "fft_simple.h"// 使用自己写的简易2d fft z->complex<double>

using namespace std;
const int N = 128;// 需要是2的幂 不然递归调用整除不完 256
const int Size = N * N;
const int N_k = N / 2 + 1;
const double dx = M_PI / 4.0;// 空间步长
const double dt = 0.01;// 减小时间步长保证稳定性（原0.01）
const int steps = 5001;
const double epsilon = 4.0 / 15.0;// 控制参数

const double psi_c = 0;// 均匀相背景值
const double A_s = 0.3;// 条纹相振幅
const double A_h = 4.0 / 5.0 * sqrt((15 * epsilon - 36 * psi_c * psi_c)/ 3.0);// 六方相振幅 -0.326

const double q0 = 1.0;// 特征波数 2pi/T
const double q_s = 2.0 * sqrt(epsilon - 3 * psi_c * psi_c);// 条纹相 0.562
const double q_h = sqrt(3.0) * q0 / 2.0;// 六方相 0.8660
const int frequency = 1000;// 输出频率

// Seed：中心坐标、取向角(弧度)、半径
struct Seed {
    int x, y;
    double theta;
    double radius;
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

void output_vtk(const vector<double>& psi, int N, const string& filename) {
    ofstream fout_vtk(filename);
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
    vector<double> psi(Size);// 实空间 psi
    vector<double> nl_R(Size);// 实空间 非线性项(nonlinear) psi^3

    // 初始化seeds
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
                
                //仅半径内添加六方相，移除外部5像素扰动
                if(dist < s.radius) val += hex_phase(dx_seed, dy_seed, s.theta, A_h);
            }

            psi[idx] = val;
        }
    } 

    // 模拟
    for (int t = 0; t < steps; ++t) {
        /*------------------------- 1.填充nl: psi^3 ------------------------------*/
        for (int i = 0; i < Size; ++i) {nl_R[i] = psi[i] * psi[i] * psi[i];}

        /*--------------------------- 2.2d fft ------------------------------*/
        vector<vector<z>> psi_k = fft_2d(psi, N);
        vector<vector<z>> nl_k = fft_2d(nl_R, N);

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
                psi_k[i][j] = (psi_k[i][j] - dt * z(k2, 0) * nl_k[i][j]) / factor2;
            } 
        }

        /*---------------- 4.ifft psi_k -> psi ----------------*/
        psi = ifft_2d(psi_k, N);   

        /*------------------ 5.输出vtk ----------------------*/
        if (t % frequency == 0) {
            cout << "step: " << t << " / " << steps << endl;
            output_vtk(psi, N, "output_" + to_string(t) + ".vtk");
        }
    }

    cout << "模拟完成！" << endl;
    return 0;
}