/*
2025.7 根据 ELder等人2002年发表在prl的文章，复现原文的pfc代码，比较简略
使用自己写的简化fft.h
clang++ -o pfc main.cpp -std=c++17
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>// 配合 srand(time(NULL)) 设置随机数种子
#include <fstream> 
#include <complex> 
#include "fft_simple.h" // 使用上一级目录的简易2d fft

using namespace std;
const int N = 256;// 需要是2的幂 不然递归调用整除不完 256
const int Size = N * N;
const int N_k = N / 2 + 1;
const double dx = M_PI / 4.0;// 空间步长
const double dt = 0.01;// 时间步长
const int steps = 100;// 时间步
const double epsilon = 4.0 / 15.0;// 控制参数 0.2667

const double psi_c = 0.25;// 均匀相背景值
const double A_s = -0.3;// 条纹相振幅
const double A_h = -4.0 / 5.0 * sqrt((15 * epsilon - 36 * psi_c * psi_c)/ 3.0);// 六方相振幅 -0.326

const double q0 = 1.0;// 特征波数 2pi/T
const double q_s = 2.0 * sqrt(epsilon - 3 * psi_c * psi_c);// 条纹相 0.562
const double q_h = sqrt(3.0) * q0 / 2.0;// 六方相 0.8660
const int frequency = 10;// 输出频率


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
    
    //输出psi
    for (int i = 0; i < N * N; ++i) {fout_vtk << psi[i] << "\n";}

    fout_vtk.close();
    cout << "Output file: " << filename << endl;
}

int main() {
    // 初始化
    vector<double> psi(Size); // 实空间 psi
    vector<double> nl_R(Size); // 实空间 psi^3
    //srand(time(NULL)); // 设置随机数种子 和当前系统时间配合 每次运行产生的随机数列不一样
    //for (int i = 0; i < Size; ++i) {psi[i] = 0.01 * (2.0 * rand() / RAND_MAX - 1.0); }// 随机初始化原子密度波序参量噪声（模拟涨落）范围控制在 [-0.01, 0.01]) 

    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < N; ++j) { 
            double x = (double)j * dx;
            double y = (double)i * dx;
            int idx = i * N + j;

            // 在六方相基础上加入微小随机涨落（±0.01）
            //psi[idx] = A_s * sin(q_s * x) + psi_c;// 条纹相 psi_s
            psi[idx] = A_h * (cos(q_h * x) * cos(q_h * y / sqrt(3.0)) + 0.5 * cos(2 * q_h * y / sqrt(3.0))) + psi_c;// 六方相 psi_h
            //psi[idx] = psi_c;// 均匀相 psi_c
        }
    } 

    // 模拟
    for (int t = 0; t < steps; ++t) {
        // 1.填充nl: psi^3
        for (int i = 0; i < Size; ++i) {nl_R[i] = psi[i] * psi[i] * psi[i];}

        // 2. 2d fft
        vector<vector<z>> psi_k = fft_2d(psi, N);
        vector<vector<z>> nl_k = fft_2d(nl_R, N);

        // 3. Fourier空间 更新psi_k（频域）
        for (int i = 0; i < N; ++i) {
            double ky = (i <= N / 2) ? i : (i - N);// 计算当前行对应波数ky 得到正负波数（频率）
            ky *= 2.0 * M_PI / (N * dx);// 从1开始计算的正波数 分配给ky

            // 遍历每一列
            for (int j = 0; j < N_k; ++j) {//列只有一半 kx只取正值（含0）
                double kx = j;
                kx *= 2.0 * M_PI / (N * dx);
                
                double k2 = kx * kx + ky * ky;
                double factor1 = q0 * q0 - k2;// q0 特征波数：晶体点阵的倒空间矢量长度
                double Lk = -k2 * (factor1 * factor1 - epsilon);

                // 半隐式欧拉法
                // psi_k_new = (psi_k_old - dt * k2 * nl_k_old) / (1 - dt * Lk)
                z factor2 = 1.0 - dt * z(Lk, 0);
                //psi_k[i][j] = psi_k[i][j] / factor2;
                psi_k[i][j] = (psi_k[i][j] - dt * z(k2, 0) * nl_k[i][j]) / factor2;
            } 
        }

        // 4. ifft psi_k -> psi
        psi = ifft_2d(psi_k, N);   

        if (t % frequency == 0) cout << "step: " << t << " / " << steps << endl;
    }

    output_vtk(psi, N, "output.vtk");
    cout << "模拟完成！" << endl;

    return 0;
}