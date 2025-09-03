/*
2025.8.14
v2 迭代
1.位反转重排 迭代
2.添加timer计时器类
【快速傅里叶变换(FFT)feat.多项式乘法 | Reducible】https://www.bilibili.com/video/BV1ee411c7B7?vd_source=cbc056bc6bdd1ffea281d8c20f903819
*/

#ifndef FFT_V2_H
#define FFT_V2_H

#include <vector>
#include <complex>
#include <cmath>

using namespace std;
using z = complex<double>;// 复数z 实部虚部都是 double 精度

/*--------------------1d fft（迭代）----------------*/
void fft_1d(vector<z>& P) {
    int N = P.size();
    
    // 1.log2_n计算 -> 确定fft层数：计算满足2^log2_n >= n的最小log_n值
    int log_n = 0;
    while ( (1 << log_n) < N ) log_n++;// 1<<log_n 二进制数左移一位 -> 2^log_n
    
    // 2.位反转重排（核心准备）
    // 0,1,2,3,4,5,6,7 → 0,4,2,6,1,5,3,7
    for (int i = 0; i < N; ++i) { 
        int reverse = 0;
        for (int j = 0; j < log_n; ++j) {
            if (i & (1 << j))  reverse |= (1 << (log_n - 1 - j));
        }// 检查i的二进制数第j位是否为1 // 如果第j位为1，则在reverse的第(log_n-1-j)位设置为1
        if (i < reverse) { swap(P[i], P[reverse]); }// 防止重复交换
    }
    
    
    // 3.fft核心 自底向上的蝶形运算
    // 外层循环 控制层级n = 2，4，8...N
    for (int n = 2; n <= N; n <<= 1) {// k=k<<1= k*2
        double theta = 2.0 * M_PI / n;// 角度增量（负值 前向FFT）
        z w_n = polar(1.0, -theta);// 第n个单位根 w_n = e^(-2πi/n)
        
        // 中层循环 N个数据分成若干长度为 n 的数组，对每组进行n点fft
        for (int i = 0; i < N; i += n) {// i += n 跳到下一个子组
            z w_k(1.0, 0.0);  // 初始化为1+0i

            // 内层循环 蝶形运算
            for (int j = 0; j < n / 2; ++j) {
                z u = P[i + j];// i 子组索引 j 组内偏移
                z v = w_k * P[i + j + n / 2];
                P[i + j] = u + v;
                P[i + j + n / 2] = u - v;
                w_k *= w_n;
            }
        }
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
    const int N_k = N / 2 + 1; // 实数FFT只需要一半频谱
    vector<vector<z>> output(N, vector<z>(N_k));
    
    // 1.做1d_x fft - 将row数组创建移到循环外
    vector<z> row(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) { row[j] = z(input[i * N + j], 0.0); }
        fft_1d(row);
        for (int j = 0; j < N_k; ++j) { output[i][j] = row[j]; }
    }
    
    // 2.做1d_y fft - 将col数组创建移到循环外
    vector<z> col(N);
    for (int j = 0; j < N_k; ++j) {
        for (int i = 0; i < N; ++i) { col[i] = output[i][j]; }
        fft_1d(col);
        for (int i = 0; i < N; ++i) { output[i][j] = col[i]; }
    }
    
    return output;
}

/*-------------------2d ifft ：输入频域复数，返回实数---------------*/
vector<double> ifft_2d(const vector<vector<z>>& input, int N) {
    const int N_k = N / 2 + 1;
    vector<vector<z>> temp(N, vector<z>(N));
    
    // 1. 还原全部频谱
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N_k; ++j) {
            temp[i][j] = input[i][j]; // 复制左半部分频谱
            if (j > 0 && j < N_k - 1) {// DC分量和奈奎斯特频率是实数不需要共轭 防止覆盖已有的重要频率
                temp[(i == 0) ? 0 : N - i][N - j] = conj(input[i][j]); // 利用共轭对称性填充右半部分
            }
        }
    }
    
    // 2.每列 ifft
    vector<z> col(N);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) { col[i] = temp[i][j]; }
        ifft_1d(col);
        for (int i = 0; i < N; ++i) { temp[i][j] = col[i]; }
    }
    
    // 3.每行 ifft，取实部作为最终输出
    vector<double> output(N * N);
    for (int i = 0; i < N; ++i) {
        ifft_1d(temp[i]);
        for (int j = 0; j < N; ++j) { output[i * N + j] = temp[i][j].real(); }// 取实部 
    }

    return output;
}

#endif