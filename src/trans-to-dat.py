import os
from PIL import Image
import threading
import sys
import subprocess
import time
import utils

def get_pixel_colors(image_path): # 读取图片数据
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    pixel_colors = []
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
            pixel_colors.append((r, g, b, a))

    return ((width, height), pixel_colors)

def process(filename):
    print("[%f] processing for %s [    ]" % (time.time(), filename))

    input_file  = os.path.join(utils.get_img_dir(), filename)
    output_file = os.path.join(utils.get_dat_dir(), (filename + ".txt").replace("png", "jpg"))

    (w, h), pixels = get_pixel_colors(input_file)
    with open(output_file, "w") as f:
        f.write("%d %d\n" % (w, h))

        for r, g, b, a in pixels:
            f.write("%d %d %d %d\n" % (r, g, b, a))

    print("[%f] processing for %s [DONE]" % (time.time(), filename))

cnt = 0

def process_in_process(file): # 在子进程中处理问题
    global cnt
    result = subprocess.run(["python", __file__, file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cnt += 1
    print("result for %s is %s (finish cnt = %d)" % (file, str(result.returncode), cnt))

def process_all(): # 有一说一这个处理过程好慢啊，干脆这一步就上 GPU 算了
    img_name_list = utils.get_all_img()

    pool = []
    for file in img_name_list:
        thread = threading.Thread(target=process_in_process, args=(file, ))
        pool.append(thread)
        thread.start()

    for thread in pool: # 等待所有线程结束
        thread.join()

if __name__ == "__main__":
    argv = utils.deepcopy(sys.argv[1:])
    
    if len(argv) == 0: # 原始进程
        process_all()
    else:
        filename = argv[0]
        process(filename) # 处理一个文件