import subprocess
import threading
from PIL import Image
import time
import os

import utils
import sys

def save_pic(filename, width, height, pixel_list_rgba):
    image = Image.new("RGBA", (width, height))
    image.putdata(pixel_list_rgba)
    image.save(os.path.join(utils.get_blr_dir(), filename))

def read_blr_file(filename):
    assert filename[-4:] == ".blr" # 检查后缀名

    with open(filename) as fpin:
        width, height = [int(x) for x in fpin.readline().split()]

        pixel_list = []
        for _ in range(width):
            column = []
            for _ in range(height):
                r, g, b, a = [int(x) for x in fpin.readline().split()]
                column.append((r, g, b, a))
            pixel_list.append(column)
    
    pixel_list = utils.transpose(pixel_list)
    pixel_list = utils.squeeze(pixel_list)

    return width, height, pixel_list

def get_all_blr_file():
    arr = []
    for file in os.listdir(utils.get_dat_dir()):
        if file[-4:] == ".blr":
            arr.append(file)
    return arr

def thread_function(file):
    print("[%f] revert to png for %s [    ]" % (time.time(), file))
    result = subprocess.run(["python", __file__, file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("[%f] revert to png for %s [DONE] %s" % (time.time(), file, str(result.returncode)))

def process_all():
    blr_list = get_all_blr_file()

    pool = []
    for file in blr_list:
        thread = threading.Thread(target=thread_function, args=(file, ))
        thread.start()
        pool.append(thread)

    for thread in pool:
        thread.join() # 汇聚

def process_for_file(filename): # 这里面要文件名，不要路径
    input_filepath  = os.path.join(utils.get_dat_dir(), filename)
    output_filepath = os.path.join(utils.get_blr_dir(), filename.replace(".jpg.txt.blr", ".png"))
    width, height, pixel_list = read_blr_file(input_filepath)
    save_pic(output_filepath, width, height, pixel_list)

if __name__ == "__main__":
    argv = utils.deepcopy(sys.argv[1:])

    if len(argv) == 0:
        process_all()
    else:
        filename = argv[0]
        process_for_file(filename)
    