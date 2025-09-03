/*
2025.8 PFC-RG晶粒生长(基于2006 Renormalization Group方法 DOI: 10.1007/s10955-005-9013-7) 性能优化版
1.常量编译时计算
2.每次update中都要计算网格坐标和相位因子 -> 预计算网格坐标和相位因子
3.main实空间计算nl 每次循环都重复计算norm()和conj() -> 预计算x
4.fft_2d 中数据类型复制成为实部虚部 -> 直接转换？x
clang++ -O3 -o apfc main.cpp -std=c++17 -I../../FFT/include -L../../FFT/lib -lfftw3
g++ -std=c++17 -O3 main.cpp -o apfc \
    -I$HOME/fftw/include \
    -L$HOME/fftw/lib \
    -lfftw3 -lm
export LD_LIBRARY_PATH=$HOME/fftw/lib:$LD_LIBRARY_PATH
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream> 
#include <complex> 
#include <filesystem>
#include <sstream>
#include "timer.h" // timer类计时
#include <fftw3.h> // 使用fftw库

using namespace std;
using z = complex<double>;
using fftw_z = fftw_complex;

inline double dot_product(z k, z x) {
    return k.real() * x.real() + k.imag() * x.imag();
}

/*------------------------- class ----------------------------*/ 
class InputPara{
public:
    int N;
    double dx;
    double dt; // 时间步长
    int steps; // 总步数
    int freq; // 输出频率

    double psi_c; // 平均密度，Fig.1
    double r; // 参数r
    double k0; // 倒格矢模长，已归一化
    double A0;

    // 构造函数，设置默认值
    InputPara() : N(128), dx(M_PI / 2.0), dt(0.04), steps(10000), freq(1000), 
                   psi_c(0.285), r(-0.25), k0(1.0), A0(0.1) {}

    // 打印参数
    void print() const {
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
        cout << "========================" << endl;
    }
};

class PreCal{
public:
    int Size;
    double Gamma; // 扰动增长率计算的常数计算Γ
    double factor_k; // 基本波矢单位
    double psi_c_6; // 预计算常量

    z k1, k2, k3;
    vector<z> k_vec;
    
    // 构造函数，根据输入参数计算预计算参数
    PreCal(const InputPara& params) {
        Size = params.N * params.N;
        Gamma = -(params.r + 3 * params.psi_c * params.psi_c);
        factor_k = 2.0 * M_PI / (params.N * params.dx);
        psi_c_6 = 6.0 * params.psi_c;
    }
};

class Fields {
public:
    fftw_plan plan_forward, plan_backward;
    fftw_z *in_z, *out_z;
    fftw_z *A_k[3], *nl_k[3];
    vector<vector<z>> A_R; 
    vector<vector<z>> nl_R; 
    
    /* 构造函数 初始化 */ 
    Fields(const InputPara& params, const PreCal& pre) : 
        A_R(3, vector<z>(pre.Size)), nl_R(3, vector<z>(pre.Size)) 
    {
        // 分配FFT内存
        in_z = (fftw_z*) fftw_malloc(sizeof(fftw_z) * pre.Size);
        out_z = (fftw_z*) fftw_malloc(sizeof(fftw_z) * pre.Size);
        for(int i = 0; i < 3; ++i) {
            A_k[i] = (fftw_z*) fftw_malloc(sizeof(fftw_z) * pre.Size);
            nl_k[i] = (fftw_z*) fftw_malloc(sizeof(fftw_z) * pre.Size);
        }
        
        // after allocations
        memset(in_z,  0, sizeof(fftw_complex) * pre.Size);
        memset(out_z, 0, sizeof(fftw_complex) * pre.Size);
        for (int i=0;i<3;++i){
            memset(A_k[i],  0, sizeof(fftw_complex) * pre.Size);
            memset(nl_k[i], 0, sizeof(fftw_complex) * pre.Size);
        }
        
        // 初始化fft Plan
        plan_forward = fftw_plan_dft_2d(params.N, params.N, in_z, out_z, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftw_plan_dft_2d(params.N, params.N, in_z, out_z, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    
    // 析构函数
    ~Fields() {
        fftw_destroy_plan(plan_forward);
        fftw_destroy_plan(plan_backward);
        fftw_free(in_z); fftw_free(out_z);
        for(int i = 0; i < 3; ++i) {
            fftw_free(A_k[i]); fftw_free(nl_k[i]);
        }
        fftw_cleanup();
    }
};

/*---------------------------------- fft ----------------------------------*/
// fft  数据类型转换 z->fftw_z
void fft_2d(const vector<z>& input, fftw_z* output, Fields& fields) {
    for (int i = 0; i < input.size(); ++i) {
        fields.in_z[i][0] = input[i].real();
        fields.in_z[i][1] = input[i].imag();
    }
    fftw_execute_dft(fields.plan_forward, fields.in_z, output);
}

// ifft  数据类型转换 fftw_z-> z  
vector<z> ifft_2d(fftw_z* input, Fields& fields, int Size) {
    fftw_execute_dft(fields.plan_backward, input, fields.out_z);
    vector<z> output(Size);
    for(int i = 0; i < Size; ++i) 
        output[i] = z(fields.out_z[i][0], fields.out_z[i][1]) / static_cast<double>(Size);
    return output;
}

/*-------------------------- 功能性函数 -----------------------------*/
// 从配置文件读取参数
void readInput(const string& filename, InputPara& params) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cout << "无法打开配置文件: " << filename << "，使用默认参数" << endl;
        return;
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
    fout_vtk << "SCALARS psi double\nLOOKUP_TABLE default\n";
    
    for (int i = 0; i < N * N; ++i) { fout_vtk << psi[i] << "\n"; }

    fout_vtk.close();
    cout << "Output file: " << filename << endl;
}

/*-------------------------- 主要函数 -----------------------------*/
// 频域更新
void update_k(fftw_z* Aj, const fftw_z* nl, const z& k_j, const InputPara& params, const PreCal& pre) {
    for (int i = 0; i < params.N; ++i) {
        double ky = (i < params.N / 2) ? i * pre.factor_k : (i - params.N) * pre.factor_k;// 正负频率
        for (int j = 0; j < params.N; ++j) {
            double kx = (j < params.N / 2) ? j * pre.factor_k : (j - params.N) * pre.factor_k;// 正负频率
            int idx = i * params.N + j;

            z q0 = { kx, ky };// 当前波矢，用复数储存二维向量！！
            z q1 = q0 + k_j;// 经过晶粒波矢k_j移位后的波矢
            double op_2 = -(dot_product(q0, q0) + 2 * dot_product(k_j, q0));// 算子平方项

            // 旋转协变 线性算子Lj
            double Lj = (dot_product(q1, q1)) * (pre.Gamma - op_2 * op_2);

            // 时间步进 半隐式欧拉法
            z Aj_k(Aj[idx][0], Aj[idx][1]);
            z nl_k_val(nl[idx][0], nl[idx][1]);// 非线性项数据 在main中显式处理 这里只是传递数据
            z Aj_new = (Aj_k + params.dt * nl_k_val) / (1. - params.dt * Lj);// 复数运算 隐式处理
            
            // 计算 Aj_new 之后立即检查
            if (!std::isfinite(Aj_new.real()) || !std::isfinite(Aj_new.imag())) {
                std::cerr << "Non-finite at idx=" << idx << " kx=" << kx << " ky=" << ky 
                        << " Lj=" << Lj << " Aj_k=" << Aj_k << " nl_k=" << nl_k_val << "\n";
                abort();
            }
            
            Aj[idx][0] = Aj_new.real();
            Aj[idx][1] = Aj_new.imag();
        }
    }
}

// 初始化函数
void initialize(Fields& fields, const InputPara& params, PreCal& pre) {
    // 初始化倒格矢
    pre.k1 = z(-sqrt(3.0) / 2.0, -0.5);// 使用复数 用来储存二维数组
    pre.k2 = z(0.0, 1.0);
    pre.k3 = z(sqrt(3.0) / 2.0, -0.5);
    pre.k_vec = { pre.k1, pre.k2, pre.k3 };
    
    // 初始化种子
    struct Seed { int x, y; double theta; double radius; };
    vector<Seed> seeds = {
        { params.N/4, params.N/4, 0, 15 },
        { 3*params.N/4, params.N/4, M_PI/6, 15 },
        { params.N/2, 3*params.N/4, M_PI/3, 15 }};
    
    for (int i = 0; i < params.N; ++i) {
        for (int j = 0; j < params.N; ++j) {
            int idx = i * params.N + j;
            fields.A_R[0][idx] = fields.A_R[1][idx] = fields.A_R[2][idx] = {0.0, 0.0}; // 默认液相
            z x_vec = z(j * params.dx, i * params.dx); // 网格坐标

            // 判断是否在晶粒内部 半径内激活初始波形 
            for (const Seed& s : seeds) {
                double dx_s = (j - s.x) * params.dx;// 到晶粒中心距离
                double dy_s = (i - s.y) * params.dx;
                double distance = sqrt(dx_s * dx_s + dy_s * dy_s);
                
                if (distance < s.radius) {
                    double cos_theta = cos(s.theta);
                    double sin_theta = sin(s.theta);
                    z k1_rot = {pre.k1.real() * cos_theta - pre.k1.imag() * sin_theta,
                                pre.k1.real() * sin_theta + pre.k1.imag() * cos_theta};
                    z k2_rot = {pre.k2.real() * cos_theta - pre.k2.imag() * sin_theta,
                                pre.k2.real() * sin_theta + pre.k2.imag() * cos_theta};
                    z k3_rot = {pre.k3.real() * cos_theta - pre.k3.imag() * sin_theta,
                                pre.k3.real() * sin_theta + pre.k3.imag() * cos_theta};

                    fields.A_R[0][idx] = params.A0 * exp(z(0, dot_product(k1_rot - pre.k1, x_vec)));
                    fields.A_R[1][idx] = params.A0 * exp(z(0, dot_product(k2_rot - pre.k2, x_vec)));
                    //fields.A_R[2][idx] = params.A0 * exp(z(0, dot_product(k3_rot - pre.k3, x_vec)));
                }
            }
        }
    }
}

// 重构psi 输出vtk
void reconstruct_output(int step, const InputPara& params, const PreCal& pre, const Fields& fields) {
    vector<double> psi_A(pre.Size);
    for (int i = 0; i < params.N; ++i) {
        for (int j = 0; j < params.N; ++j) {
            int idx = i * params.N + j;
            z x_vec = { j * params.dx, i * params.dx };// 复数存储 当前网格点空间物理位置向量坐标

            z A_sum = fields.A_R[0][idx] * exp(z(0, dot_product(pre.k_vec[0], x_vec))) +
                      fields.A_R[1][idx] * exp(z(0, dot_product(pre.k_vec[1], x_vec))) +
                      fields.A_R[2][idx] * exp(z(0, dot_product(pre.k_vec[2], x_vec)));
            psi_A[idx] = params.psi_c + 2.0 * A_sum.real();
        }
    }
    // 只有每freq步会把线性项和非线性项合并并转化为psi 并输出
    output_vtk(psi_A, params.N, "output_" + to_string(step) + ".vtk");
}

int main() {
    InputPara params;
    readInput("input.txt", params);
    params.print();
    PreCal pre(params);
    Fields fields(params, pre);
    Timer timer(params.steps, params.freq);
    
    // 初始化 开始计时
    initialize(fields, params, pre);
    timer.start();

    // 模拟
    for (int t = 1; t < params.steps; ++t) {
        /*------------------ 1.实空间计算非线性项 N_j ------------------------------*/
        for (int i = 0; i < pre.Size; ++i) {
            double A_abs2[3] = { norm(fields.A_R[0][i]), 
                                 norm(fields.A_R[1][i]), 
                                 norm(fields.A_R[2][i]) };
            z A_conj[3] = { conj(fields.A_R[0][i]), 
                            conj(fields.A_R[1][i]), 
                            conj(fields.A_R[2][i]) };
            
            fields.nl_R[0][i] = -3.0 * fields.A_R[0][i] * (A_abs2[0] + 2.0 * (A_abs2[1] + A_abs2[2])) - 
                                pre.psi_c_6 * A_conj[1] * A_conj[2];
            fields.nl_R[1][i] = -3.0 * fields.A_R[1][i] * (A_abs2[1] + 2.0 * (A_abs2[0] + A_abs2[2])) - 
                                pre.psi_c_6 * A_conj[0] * A_conj[2];
            fields.nl_R[2][i] = -3.0 * fields.A_R[2][i] * (A_abs2[2] + 2.0 * (A_abs2[0] + A_abs2[1])) - 
                                pre.psi_c_6 * A_conj[0] * A_conj[1];
        }

        /*--------------------------- 2.fft和ifft k空间更新 ------------------------------*/
        for (int j = 0; j < 3; ++j) {
             fft_2d(fields.A_R[j], fields.A_k[j], fields);
             fft_2d(fields.nl_R[j], fields.nl_k[j], fields);
             update_k(fields.A_k[j], fields.nl_k[j], pre.k_vec[j], params, pre);
             fields.A_R[j] = ifft_2d(fields.A_k[j], fields, pre.Size);
         }

        /*------------------ 3.根据freq 重构psi 输出vtk ----------------------*/
        if (t % params.freq == 0 || t == 0) {
            timer.update(t);
            reconstruct_output(t, params, pre, fields);
        }
    }

    timer.finish();
    cout << "模拟完成！" << endl;
    
    return 0;
}