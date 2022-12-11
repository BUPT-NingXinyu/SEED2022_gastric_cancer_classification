import os
import json
import numpy as np
import glob
import kfbReader
import cv2
from tqdm import tqdm
import random
from PIL import Image

def json2point_npy(path, save_path):
    file_list = os.listdir(path)
    for json_name in file_list:
        tem_json_name = json_name.split('.')[0]
        j_path = os.path.join(path, json_name)
        f = open(j_path, 'r', encoding='utf-8')
        annotation = json.load(f)
        for idx, item in enumerate(annotation['contexts']):
            tem_list = []
            for coordinate in item['points']:
                tem_list.append([coordinate['x'], coordinate['y']])
            if not os.path.exists(os.path.join(save_path, tem_json_name)):
                os.makedirs(os.path.join(save_path, tem_json_name))
            region = np.array(tem_list)
            np.save(os.path.join(save_path, tem_json_name, f"{tem_json_name}_{idx}"), region)

# json标注生成npy文件
def generate_npy():
    paths = [[r'/dataset/cancer/train/含T1标注的json',r'/tmp/npy/T1'],
         [r'/dataset/cancer/train/含T2标注的json',r'/tmp/npy/T2'],
         [r'/dataset/cancer/train/含T3标注的json',r'/tmp/npy/T3'],
         [r'/dataset/cancer/train/含Tis标注的json',r'/tmp/npy/Tis'],
         [r'/dataset/cancer/test/json',r'/tmp/npy/Test'],
        ]

    for i in range(len(paths)):
        json_path, point_path = paths[i]
        json2point_npy(json_path, point_path)

# T1,T2,T3,Tis生成patch
def generate_T1_2_3_is_patch(patch_number_of_one_anno, scale):
    dir_list = [['T1','含T1区域的kfb'],
        ['T2','含T2区域的kfb'],
        ['T3','含T3区域的kfb'],
        ['Tis','含Tis区域的kfb']]
    #level = 2  # openslide
    #scale = 2  # kfbreader
    edge = 512

    for temp in dir_list:
        wsi_path = f'/dataset/cancer/train/{temp[1]}'
        patch_path = f'/tmp/train_data/{temp[0]}_patch'
        #patch_path = f'/opt/project/project/img/{temp[0]}_patch'
        ann_path = f'/tmp/npy/{temp[0]}'
        wsi_name_list = os.listdir(ann_path)
        
        print(f"processing {temp} .......")

        for i in range(len(wsi_name_list)):
            #wsi_name_list[i] = wsi_name_list[i] + '.svs'
            wsi_name_list[i] = wsi_name_list[i] + '.kfb'

        for item in tqdm(wsi_name_list):      # item 是kfb的文件名
            #print(os.path.join(wsi_path, item))
            reader = kfbReader.reader()
            reader.ReadInfo(os.path.join(wsi_path, item), scale=0)
            raw_w = reader.getWidth()
            raw_h = reader.getHeight()


            reader.ReadInfo(os.path.join(wsi_path, item), scale=scale)
            W = reader.getWidth()
            H = reader.getHeight()

            down_sample_rate = int(raw_w/W)
            judge_edge = int(edge/down_sample_rate)
            
            drop_suffix = item.split('.')[0]
            save_path = os.path.join(patch_path, drop_suffix)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            npy_path = glob.glob(os.path.join(ann_path, drop_suffix, '*.npy'))
            region_num = len(npy_path)
            region_points = []
            # 一次取一个标注区域，在该区域中取 50 个 patch
            anno_num = 0
            for it in npy_path:
                region_points = [(np.load(it)/down_sample_rate).astype(np.int32)]
                region_points = np.array(region_points)

                mask = np.zeros((H, W))
                cv2.fillPoly(mask, region_points, 255)
                x, y = np.where(mask)
                all_point = np.stack(np.vstack((x, y)), axis=1)
                sample_num = 500
                if all_point.shape[0] > sample_num:
                    sample_point = all_point[np.random.randint(all_point.shape[0], size=sample_num), :]
                else:
                    sample_point = all_point
                save_num = 0
                for idx, (x, y) in enumerate(sample_point):
                    # print('-------------------------')
                    # print(f'{drop_suffix}_Anno{anno_num}_{idx}')
                    # print('edge',edge)
                    # print('y',y)
                    # print('W',W)
                    # print('x',x)
                    # print('H',H)
                    # print('-------------------------')
                    if save_num < patch_number_of_one_anno:
                        if mask[max(x-judge_edge,0), y] == 0 or \
                        mask[x, max(y-judge_edge,0)] == 0 or \
                        mask[max(x-judge_edge,0), max(y-judge_edge,0)] == 0 or \
                        mask[min(x+judge_edge,H-1), y] == 0 or\
                        mask[min(x+judge_edge,H-1), min(y+judge_edge,W-1)] == 0 or \
                        mask[x, min(y+judge_edge,W-1)] == 0:
                            continue
                        else:
                            img = reader.ReadRoi((int(np.max((y-edge/2),0))), int(np.max((x-edge/2),0)), edge,edge, scale)
                            # 有色区域大于一定比例的图片保留
                            img_gray = cv2.cvtColor(np.array(img)[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                            ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
                            color_area = len(np.where(thresh)[0])
                            if(color_area>0.1*edge*edge and color_area<0.95*edge*edge):
                                cv2.imwrite(os.path.join(save_path, f'{drop_suffix}_Anno{anno_num}_{idx}.png'),img)
                                save_num += 1
                    else:
                        break
                                
                anno_num += 1

# T0类生成patch
def generate_T0_patch(patch_number_of_one_anno, scale):
    path = r'/dataset/cancer/train/T0'
    svs_file = os.listdir(path)
    save = r'/tmp/train_data/T0_patch'
    #save = r'/opt/project/project/img/T0_patch'

    #level = 2
    #scale =2  # kfbreader
    edge = 512

    # 读取到边界时，将边界以外的透明区域转换为白色。
    def transparence2white(img):  
        sp = img.shape  # 获取图片维度
        width = sp[0]  # 宽度
        height = sp[1]  # 高度
        for yh in range(height):
            for xw in range(width):
                color_d = img[xw, yh]  # 遍历图像每一个点，获取到每个点4通道的颜色数据
                if (color_d[3] == 0):  # 最后一个通道为透明度，如果其值为0，即图像是透明
                    img[xw, yh] = [255, 255, 255, 255]  # 则将当前点的颜色设置为白色，且图像设置为不透明
        return img

    for svs in tqdm(svs_file):
        svs_drop_f = svs.split('.')[0]
        svs_path = os.path.join(path, svs)
        #slide = openslide.OpenSlide(svs_path)
        # if len(slide.level_downsamples)<=level:
        #     level = len(slide.level_downsamples)-1
        # else:
        #     level = 2
        reader = kfbReader.reader()
        reader.ReadInfo(svs_path, scale=0)
        raw_w = reader.getWidth()
        raw_h = reader.getHeight()


        reader.ReadInfo(svs_path, scale=scale)
        W = reader.getWidth()
        H = reader.getHeight()

        down_sample_rate = int(raw_w/W)
        judge_edge = int(edge/down_sample_rate)
        
        #print("kk", svs_drop_f)
        #print(H, W)
        list_x = [i for i in range(H)]
        list_y = [i for i in range(H)]
        random.shuffle(list_x)
        random.shuffle(list_y)
        save_path = os.path.join(save, svs_drop_f)
        #print('debug')
        #print(svs_drop_f)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_num = 0
        for idx in range(500):
            if save_num<patch_number_of_one_anno:
                if list_x[idx] + judge_edge > H or list_y[idx] + judge_edge> W:
                    continue
                else:
                    img = reader.ReadRoi((int(np.max((list_y[idx]-edge/2),0))), int(np.max((list_x[idx]-edge/2),0)), edge,edge, scale)
                    # 有色区域大于一定比例的图片保留
                    np_img = np.array(img)[:, :, :3].astype(np.uint8)
                    #transparence2white(np_img)
                    #np_img = np_img[:, :, :3]
                    img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
                    # cv2.imshow('img_gray',img_gray)
                    # cv2.imshow('thresh',thresh)
                    # cv2.waitKey(0)
                    # print(len(np.where(thresh)[0]))
                    # print('0.1*edge*edge',0.1*edge*edge)
                    color_area = len(np.where(thresh)[0])
                    if(color_area>0.1*edge*edge and color_area<0.95*edge*edge):
                        cv2.imwrite(os.path.join(save_path, f'{svs_drop_f}_{save_num}.png'),img)
                        save_num += 1
            else:
                break


# Test类生成patch
def generate_Test_patch(patch_number_of_one_anno, scale):
    svs_path = r'/dataset/cancer/test/kfb'
    root_path = r'/tmp/npy/Test'
    save_p = r'/tmp/Test_patch'
    #save_p = r'/opt/project/project/img/Test_patch'

    #level = 2
    #scale =2  # kfbreader
    edge = 512
    sample_num = 500

    file_list = os.listdir(root_path)
    for file_name in tqdm(file_list):
        file_path = os.path.join(root_path, file_name)      # F:\svs\含Test标注的npy\4mF7tL
        annotation_list = os.listdir(file_path)
        for annotation_name in annotation_list:
            annotation_path = os.path.join(file_path, annotation_name)
            save_path = os.path.join(save_p, file_name, annotation_name.split('.')[0].replace(' ', ''))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            reader = kfbReader.reader()
            reader.ReadInfo(os.path.join(svs_path, f'{file_name}.kfb'), scale=0)
            raw_w = reader.getWidth()
            raw_h = reader.getHeight()


            reader.ReadInfo(os.path.join(svs_path, f'{file_name}.kfb'), scale=scale)
            W = reader.getWidth()
            H = reader.getHeight()

            down_sample_rate = int(raw_w/W)
            judge_edge = int(edge/down_sample_rate)
            
            region_points = (np.load(annotation_path)/down_sample_rate).astype(np.int32)
            
            #W, H = slide.level_dimensions[level]
            mask = np.zeros((H, W))
            cv2.fillConvexPoly(mask, region_points, 255)
            x, y = np.where(mask)
            all_point = np.stack(np.vstack((x, y)), axis=1)
            if all_point.shape[0] == 0:
                print(annotation_name)
                continue
            sample_point = all_point[np.random.randint(all_point.shape[0], size=sample_num), :]
            save_num = 0
            for idx, (x, y) in enumerate(sample_point):
                if save_num<patch_number_of_one_anno:
                    # y + judge_edge > W or x + judge_edge > H or \
                    if mask[int(max(x-edge/8,0)), y] == 0 or \
                    mask[x, int(max(y-edge/8,0))] == 0 or \
                    mask[int(max(x-edge/8,0)), int(max(y-edge/8,0))] == 0 or\
                    mask[int(min(x+edge/8,H-1)), y] == 0 or \
                    mask[x, int(min(y+edge/8,W-1))] == 0 or \
                    mask[int(min(x+edge/8,H-1)), int(min(y+edge/8,W-1))] == 0:
                        continue
                    else:
                        img = reader.ReadRoi((int(np.max((y-edge/2),0))), int(np.max((x-edge/2),0)), edge,edge, scale)
                        # 有色区域大于一定比例的图片保留
                        np_img = np.array(img)[:, :, :3].astype(np.uint8)
                        #transparence2white(np_img)
                        #np_img = np_img[:, :, :3]
                        img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
                        color_area = len(np.where(thresh)[0])
                        if(color_area>0.1*edge*edge and color_area<0.95*edge*edge):
                            cv2.imwrite(os.path.join(save_path, f'{save_num}.png'),img)
                            save_num += 1
                else:
                    break

#文件夹合并
def merge_dir():
    sour_path = r'/tmp/train_data/'
    # save_path = ['/tmp/raw/T0',
    #             '/tmp/raw/T1', 
    #             '/tmp/raw/T2', 
    #             '/tmp/raw/T3', 
    #             '/tmp/raw/Tis']
    save_path = '/tmp/raw/'
    t_patch_path = os.listdir(sour_path)
    for idx, t_path in enumerate(t_patch_path):
        print('processing ',t_path)
        save_t_path = os.path.join(save_path, t_path.split('_')[0])
        if not os.path.exists(save_t_path):
            os.makedirs(save_t_path)
        json_path = os.path.join(sour_path, t_path)
        json_list = os.listdir(json_path)
        for item in tqdm(json_list):
            temp = os.path.join(json_path, item, '*.png')
            img_path = glob.glob(temp)
            for j, i in enumerate(img_path):
                img = Image.open(i)
                img.save(os.path.join(save_t_path, f'{item}_{j}.png'))