#include <cassert>
#include <vector>

#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>

#include "MyReader.h"

#define BLUR_R      (31)
#define HALF_BLUR_R (BLUR_R >> 1)
#define HYPO_R      (32) // 我们假设每个线程块中有 8 * 8 个线程

__global__ void blur_kernel(PixelData* gpu_memory, int width, int height) {
    int x = blockIdx.x * HYPO_R + threadIdx.x;
    int y = blockIdx.y * HYPO_R + threadIdx.y;

    if(0 <= x && x < width && 0 <= y && y < height) { // 一个有效像素
        float r = 0, g = 0, b = 0, a = 0;
        int cnt = 0;

        for(int dx = -HALF_BLUR_R; dx <= HALF_BLUR_R; dx += 1) {
            for(int dy = -HALF_BLUR_R; dy <= HALF_BLUR_R; dy += 1) {
                int nx = x + dx;
                int ny = y + dy;

                if(0 <= nx && nx < width && 0 <= ny && ny < height) {
                    cnt += 1;
                    PixelData data_now = gpu_memory[nx * height + ny];

                    r += data_now.r;
                    g += data_now.g;
                    b += data_now.b;
                    a += data_now.a;
                }
            }
        }
        __syncthreads(); // 这里可能有鲁棒性问题，因为不能保证所有线程同步
        gpu_memory[x * height + y] = (PixelData) {
            .r = (uint8_t)(r / cnt),
            .g = (uint8_t)(g / cnt),
            .b = (uint8_t)(b / cnt),
            .a = (uint8_t)(a / cnt),
        }; // 数据写回
    }
}

void make_blur(std::string filename) { // 对特定图片计算模糊
    MyReader myReader(filename);

    int pic_width  = myReader.getWidth ();
    int pic_height = myReader.getHeight();

    dim3 grid_size ((pic_width + (HYPO_R - 1))/HYPO_R, (pic_height + (HYPO_R - 1))/HYPO_R);
    dim3 block_size(HYPO_R, HYPO_R);

    // printf("[%f] kernel function: blur_kernel running [    ]\n", getSystemTime());
    blur_kernel<<<grid_size, block_size>>>(myReader.getGpuDataAddr(), pic_width, pic_height);
    cudaDeviceSynchronize();
    myReader.writeBackToCpu();
    // printf("[%f] kernel function: blur_kernel running [DONE]\n", getSystemTime());

    myReader.save();
}

int main() {
    assert(BLUR_R % 2 == 1); // 模糊半径必须是奇数
    std::vector<pid_t> pid_list;

    for(int i = 1; i <= 36; i += 1) {
        pid_t pid = fork(); // fork 一个子线程出来
        assert(pid != -1);

        if(pid == 0) {
            char str[4];
            sprintf(str, "%02d", i);

            std::string filename = (std::string)"../dat/" + str + ".jpg.txt";
            make_blur(filename);
            // puts(""); // 输出一个空行
            return 0;
        }else {
            pid_list.push_back(pid);
        }
    }

    for(auto pid: pid_list) {
        int status = 0;
        int npid   = waitpid(pid, &status, 0);
    }
    return 0;
}
