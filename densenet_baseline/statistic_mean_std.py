import os
import glob
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
""" 统计数据库中所有图片的每个通道的均值和标准差 """

if __name__ == '__main__':

    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    result_mean = []
    result_std = []
    img_num = 0
    temp = []
    if len(train_files) < 10000:
        batch_size = len(train_files)
    else:
        batch_size = len(train_files)/20
    for file in tqdm(train_files):
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255.
        if img_num<batch_size:
            temp.append(img)
            img_num += 1
        else:
            mean = np.mean(temp, axis=(0,1,2))
            std = np.std(temp, axis=(0,1,2))
            img_num = 0
            temp = []
            result_mean.append(mean)
            result_std.append(std)

    all_mean = np.mean(np.array(result_mean), axis=0)
    all_std = np.mean(np.array(result_std), axis=0)
    print(all_mean)
    print(all_std)