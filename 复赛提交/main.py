import os
import json
import numpy as np
import glob
import kfbReader
import cv2
from tqdm import tqdm
import random
from PIL import Image

# import sys
# sys.path.append("/opt/project/project/")

# print(os.getcwd())
# os.chdir('/opt/project/project/')
# print(os.getcwd())

from code.preprocess import generate_npy, generate_T1_2_3_is_patch, generate_T0_patch, generate_Test_patch, merge_dir
from code.train import train_model
from code.predict import predict_result



def main(epochs):
 
 if not os.path.exists('/opt/project/project/model/'):
        os.makedirs('/opt/project/project/model/')
 if not os.path.exists('/opt/project/project/result/'):
       os.makedirs('/opt/project/project/result/')
     
 train_model(epochs)
 
 predict_result()

if __name__ == '__main__':
 env = os.getenv('environment')
 print(env)
 env="debug"
 if env=="debug":
   #print('debug')
   #generate_npy()
   generate_T1_2_3_is_patch(patch_number_of_one_anno=1, scale=2.5)
   generate_T0_patch(patch_number_of_one_anno=1, scale=2.5)
   generate_Test_patch(patch_number_of_one_anno=1, scale=2.5)
   merge_dir()
   main(3)
 elif env=="real":
 #else:
   #print('real')
   generate_npy()
   generate_T1_2_3_is_patch(patch_number_of_one_anno=40, scale=2.5)
   generate_T0_patch(patch_number_of_one_anno=40, scale=2.5)
   generate_Test_patch(patch_number_of_one_anno=40, scale=2.5)
   merge_dir()
   main(50)