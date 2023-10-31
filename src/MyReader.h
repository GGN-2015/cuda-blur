#pragma once
#include <cstdio>
#include <cassert>
#include <string>

#include <cuda_runtime.h>
#include "Common.h"

class MyReader{
public:
    MyReader(std::string filename): m_filename(filename), m_gpu_pixelDataList(nullptr) {
        FILE* fpin = fopen(m_filename.c_str(), "r");
        assert(fpin != NULL);

        printf("[%f] reading filename = %s [    ]\n", getSystemTime(), filename.c_str());
        fscanf(fpin, "%d%d", &m_width, &m_height);           // 输入图片尺寸
        m_pixelDataList = new PixelData[m_width * m_height]; // 申请存储空间

        for(int i = 0; i < m_width * m_height; i += 1) {
            int ir, ig, ib, ia;
            fscanf(fpin, "%d%d%d%d", &ir, &ig, &ib, &ia);

            m_pixelDataList[i] = (PixelData){
                .r = (uint8_t)ir,
                .g = (uint8_t)ig,
                .b = (uint8_t)ib,
                .a = (uint8_t)ia // 把读取到的颜色填充到向量中
            };
        }
        printf("[%f] reading filename = %s [DONE]\n", getSystemTime(), filename.c_str());
        fclose(fpin);
    }
    ~MyReader() {
        delete[] m_pixelDataList; // 记得释放

        if(m_gpu_pixelDataList != nullptr) { // 自动释放
            cudaFree(m_gpu_pixelDataList);
        }
    }
    PixelData* getDataAddr() const {
        return m_pixelDataList; // 获取内存空间首地址
    }
    PixelData* getGpuDataAddr() { // 将数据自动迁移到 GPU
        if(m_gpu_pixelDataList == nullptr) {
            cudaMalloc((void**)&m_gpu_pixelDataList, this->getSize()); // 申请
            cudaMemcpy(m_gpu_pixelDataList, m_pixelDataList, this->getSize(), cudaMemcpyHostToDevice); // 拷贝
        }
        return m_gpu_pixelDataList;
    }
    int getWidth() const {
        return m_width;
    }
    int getHeight() const {
        return m_height;
    }
    long long getSize() const {
        return (long long)m_width * m_height * sizeof(PixelData);
    }
    void writeBackToCpu() { // 把 GPU 上的数据拷贝回 CPU
        cudaMemcpy(m_pixelDataList, m_gpu_pixelDataList, getSize(), cudaMemcpyDeviceToHost);
    }
    void save() const {
        std::string new_filename = m_filename + ".blr";
        FILE* fpout = fopen(new_filename.c_str(), "w");
        printf("[%f] saving data to %s [    ]\n", getSystemTime(), new_filename.c_str());

        fprintf(fpout, "%d %d\n", getWidth(), getHeight());
        for(int i = 0; i < getWidth() * getHeight(); i += 1) {
            fprintf(fpout, "%u %u %u %u\n", 
                m_pixelDataList[i].r, 
                m_pixelDataList[i].g, 
                m_pixelDataList[i].b, 
                m_pixelDataList[i].a); // 输出调整后的图片到文件
        }
        printf("[%f] saving data to %s [DONE]\n", getSystemTime(), new_filename.c_str());
        fclose(fpout);
    }

private:
    std::string   m_filename;
    PixelData*    m_pixelDataList;
    PixelData*    m_gpu_pixelDataList;
    int           m_width;
    int           m_height;
};