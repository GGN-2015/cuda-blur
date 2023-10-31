#pragma once

#include <stdint.h>
#include <ctime>

struct PixelData {
    uint8_t r, g, b, a; // 存储无符号整数
};

static inline double getSystemTime() {
    struct timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    double timestamp = (currentTime.tv_sec * 1000ll + currentTime.tv_nsec / 1000000.0);
    return timestamp / 1000.0;
}

static inline int up2pow(int value_now) {
    int val = 1;
    while(val < value_now) {
        val <<= 1;
    }
    return val; // 返回二的整数次幂，要恰好大于等于 value_now
}