import json
import os

def deepcopy(mem):
    return json.loads(json.dumps(mem))

def get_src_dir():
    dir_now = os.path.dirname(os.path.abspath(__file__))
    return dir_now

def get_root_dir():
    return os.path.dirname(get_src_dir())

def get_img_dir():
    return os.path.join(get_root_dir(), "img")

def get_dat_dir():
    return os.path.join(get_root_dir(), "dat")

def get_blr_dir():
    return os.path.join(get_root_dir(), "blr")

def get_all_img() -> list:
    arr = []
    img_dir = get_img_dir()
    for file in os.listdir(img_dir): # 枚举所有图片
        filepath = os.path.join(img_dir, file)

        if os.path.isfile(filepath):
            arr.append(file)
    return arr

def transpose(mat): # 矩阵转置
    mat = deepcopy(mat)
    ans = []
    for i in range(len(mat[0])):
        row = []
        for j in range(len(mat)):
            row.append(mat[j][i])
        ans.append(row)
    return ans

def squeeze(mat):
    mat = deepcopy(mat)
    ans = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            ans.append(tuple(mat[i][j]))
    return ans