/*
2025.7.14
用于练手的简单的FFT以代替fftw库文件，详细理解课程见bilibili：
【快速傅里叶变换(FFT)feat.多项式乘法 | Reducible】https://www.bilibili.com/video/BV1ee411c7B7?vd_source=cbc056bc6bdd1ffea281d8c20f903819
*/
#ifndef SIMPLE_FFT_H
#define SIMPLE_FFT_H

#include <complex> // 引入复数库
#include <cmath>
#include <vector>

using namespace std;
using z = complex<double>;// 复数z 实部虚部都是 double 精度

/*-------------------1d fft（递归 Cooley-Tukey）---------------*/
void fft_1d(vector<z>& P) {
    int n = P.size();
    if (n <= 1) return;// n = 1，递归终止

    // 1.分离奇数项和偶数项
    vector<z> Pe(n / 2), Po(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        Pe[i] = P[2 * i];// even number 偶数
        Po[i] = P[2 * i + 1];// odd number 奇数
    }

    // 2.递归调用
    fft_1d(Pe);
    fft_1d(Po);

    // 3.fft核心 合并结果
    double theta = 2 * M_PI / n;// 角度增量
    z w_n = polar(1.0, theta);// 旋转因子 w_n = e^(i*theta) 创建复数的极坐标形式
    z w_k(1.0, 0.0);// 初始化为1+0i 

    for (int k = 0; k < n / 2; ++k) {
        z t = w_k * Po[k];
        P[k] = Pe[k] + t;
        P[k + n / 2] = Pe[k] - t;
        w_k *= w_n;
    }
}

/*-------------------1d ifft（共轭+FFT）---------------*/ 
void ifft_1d(vector<z>& P) {
    for (z& x : P) x = conj(x);// 取共轭
    fft_1d(P);
    for (z& x : P) x = conj(x) / static_cast<double>(P.size());// 共轭并归一化 size_t转化为double
}

/*-------------------2d fft：输入实数数组返回二维复数频域（保留一半谱）---------------*/ 
vector<vector<z>> fft_2d(const vector<double>& input, int N) {
    int N_k = N / 2 + 1;// 只保留一半
    vector<vector<z>> output(N, vector<z>(N_k));

    // 1.做1d_x fft
    for (int i = 0; i < N; ++i) {
        vector<z> row(N);
        for (int j = 0; j < N; ++j) { row[j] = z(input[i * N + j], 0.0); }// 实数转复数
        fft_1d(row);
        for (int j = 0; j < N_k; ++j) { output[i][j] = row[j]; }// 仅保存前半部分
    }

    // 2.做1d_y fft
    for (int j = 0; j < N_k; ++j) {
        vector<z> col(N);
        for (int i = 0; i < N; ++i) { col[i] = output[i][j]; }
        fft_1d(col);
        for (int i = 0; i < N; ++i) { output[i][j] = col[i]; }
    }

    return output;
}

/*-------------------2d ifft ：输入频域复数，返回实数---------------*/  
vector<double> ifft_2d(const vector<vector<z>>& input, int N) {
    int N_k = N / 2 + 1;
    vector<vector<z>> temp(N, vector<z>(N));

    // 1.还原全部频谱 
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N_k; ++j) {
            temp[i][j] = input[i][j]; // 复制左半部分频谱
            if (j > 0 && j < N_k - 1) {// DC分量和奈奎斯特频率是实数不需要共轭 防止覆盖已有的重要频率
                temp[(i == 0) ? 0 : N - i][N - j] = conj(input[i][j]); // 利用共轭对称性填充右半部分
            }
        }
    }

    // 2.每列 ifft
    for (int j = 0; j < N; ++j) {
        vector<z> col(N);
        for (int i = 0; i < N; ++i) { col[i] = temp[i][j]; }
        ifft_1d(col);
        for (int i = 0; i < N; ++i) { temp[i][j] = col[i]; }
    }

    // 3.每行 ifft，取实部作为最终输出
    vector<double> output(N * N);
    for (int i = 0; i < N; ++i) {
        vector<z> row = temp[i];
        ifft_1d(row);
        for (int j = 0; j < N; ++j) { output[i * N + j] = row[j].real(); }// 取实部 
    }

    return output;
}

#endif