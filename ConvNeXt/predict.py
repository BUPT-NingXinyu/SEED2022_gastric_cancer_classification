import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import convnext_base as create_model

from tqdm import tqdm
import csv
import glob


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 5
    img_size = 512
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.799, 0.662, 0.800], [0.164, 0.214, 0.149])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    sour_path = './Test_patch'
    t_patch_path = os.listdir(sour_path)
    f = open('result.csv','w',newline='')    #以写模式打开`test.csv`
    
    with torch.no_grad():
        with f:             # with可以在程序段结束后自动close
            w = csv.writer(f,dialect="excel") 
            for idx, t_path in tqdm(enumerate(t_patch_path)):
                anno_path = os.path.join(sour_path, t_path)
                anno_list = os.listdir(anno_path)
                for item in anno_list:
                    temp = os.path.join(anno_path, item, '*.png')
                    img_paths = glob.glob(temp)
                    one_anno_all_result = []
                    for idx, img_path in enumerate(img_paths):
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = data_transform(img)
                        # expand batch dimension
                        img = torch.unsqueeze(img, dim=0)
                        # predict class
                        output = torch.squeeze(model(img.to(device))).cpu()
                        predict = torch.softmax(output, dim=0)
                        predict_cla = torch.argmax(predict).numpy()
                        predict_cla = int(predict_cla.item())
                        one_anno_all_result.append(predict_cla)
                    print(one_anno_all_result)
                    pre_result = max(set(one_anno_all_result), key = one_anno_all_result.count)
                    img_name = item.split('_')[0] + '_Annotation' + item.split('_')[1]
                    row = [img_name, pre_result]
                    w.writerow(row) #按行写入
                    

if __name__ == '__main__':
    main()
