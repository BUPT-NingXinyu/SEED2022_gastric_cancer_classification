# 2022SEED第三届江苏大数据开发与应用大赛——医疗卫生赛道——杰斯的团队

本方案获得2022SEED第三届江苏大数据开发与应用大赛——医疗卫生赛道复赛第6名，综合第7名。

初赛分类 F1 score 为 0.8807。复赛分类 F1 score 为 0.7147。

## 初赛

### 数据预处理

* kfb2svs2jpg：将kfb病理切片扫描转换为jpg图片
* svs_mask2jpg.ipynb：将svs图片按照json标注做mask保存为jpg

2022.11.18

* svs_mask2jpg.ipynb：将svs图片按照json标注做mask保存为jpg。 将 mask 后的图像与原始图像融合，融合比例 alpha 选择从 0.1 到 0.9 步长为 0.1

### 分类模型

Resnet18 作为 baseline
* 训练 notebook：resnet_baseline.ipynb
* 代码：resnet_baseline 文件夹

2022.11.18
修改notebook：resnet_baseline.ipynb 对应 train.py
* Resnet50 作为 baseline

densenet169

ConvNeXt-base

## 复赛

代码见 /复赛提交。该文件夹包括复赛提交的全部代码及说明文件，具体包括医疗影像预处理代码、模型训练代码及模型预测代码。

### 数据预处理

* 将原有使用openslide库的部分转换为使用kfbReader库实现。更详细内容见 /复赛提交/description/README.md
* 最高分模型使用 kfbReader scale 参数为1.25。

### 分类模型

ConvNeXt-base

在 main.py 中设置使用 kfbReader 裁图 scale 参数大小。每个 Annotation 的采样张数和训练轮次。

安装 requirement.txt 中相关库。

运行代码：

```
python main.py
```
