/*
 * Timer类计时器
 *   Timer timer(total_stepss, frequency);
 *   timer.start();
 *   在循环中调用 timer.update(current_stepss);
 *   timer.finish();
 */

#ifndef TIMER_H
#define TIMER_H

#include <cstdio>
#include <chrono>

class Timer {
    std::chrono::high_resolution_clock::time_point t0;
    int total, freq;
    bool run;
    
    void get_hms(double seconds, int& h, int& m, int& s) {
        h = seconds / 3600;
        m = (seconds - h * 3600) / 60;
        s = seconds - h * 3600 - m * 60;
    }
    
public:
    Timer(int total = 1000, int freq = 100) : total(total), freq(freq), run(false) {}
    
    void start() { 
        t0 = std::chrono::high_resolution_clock::now(); 
        run = true; 
    }
    
    void update(int steps) {
        if (!run || steps % freq != 0) return;
        
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0);
        double elapsed = duration.count() / 1000.0;
        double progress = (steps * 100.0) / total;
        
        // 第一个frequency时间步显示总时间估算
        if (steps == freq) {
            int est_h, est_m, est_s;
            get_hms((elapsed / steps) * total, est_h, est_m, est_s);
            printf("Total_Time: %d hours %d minutes %d seconds\n", est_h, est_m, est_s);
        }
        
        // 显示进度
        int h, m, s;
        get_hms(elapsed, h, m, s);
        printf("Steps: %d/%d (%.1f%%)\t | Time: %d hours %d minutes %d seconds\n", 
               steps, total, progress, h, m, s);
    }
    
    void finish() {
        if (!run) return;
        
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0);
        double elapsed = duration.count() / 1000.0;
        int h, m, s;
        get_hms(elapsed, h, m, s);
        printf("Simulation completed. Time: %d hours %d minutes %d seconds\n", h, m, s);
        run = false;
    }
};

#endif