# baseline
## 数据预处理
svs level 选择 3。图片大小在 1000 X 1000 左右。

根据 json 标注提取目标区域，其他区域做 Mask。

训练数据量

T0 200

T1 76

T2 87

T3 148

Tis 99

测试数据量就是提交要求的 126

![](../img/01.png)

## 数据集划分

按照 0.95 ：0.05 划分训练集和验证集

## 模型选择
resnet 18 做分类

数据集尺寸 128*128

epoch 50

batch size 4

learning rate 0.0001

weight_decay 0.0001

验证集最高准确度 0.84

提交准确度 0.14

# baseline_V2
## 数据预处理

svs level 选择 3。图片大小在 1000 X 1000 左右。

根据 json 标注提取目标区域，其他区域做 Mask。

训练数据量

T0 200

T1 76

T2 87

T3 148

Tis 99

测试数据量就是提交要求的 126

![](../img/02.png)

## 数据集划分
按照 0.95 ：0.05 划分训练集和验证集
## 模型选择
resnet 50 做分类

数据集尺寸 512*512

epoch 15

batch size 32

learning rate 0.0001

weight_decay 0.0001

提交两版：

checkpoint-15.pth

验证集最高准确度 1

提交准确度 0.4129

checkpoint-10.pth

验证集最高准确度 0.992

提交准确度 0.5979