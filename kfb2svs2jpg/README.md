# kfb2svs2jpg

将比赛提供的kfb格式数据转化为jpg图片。

* 首先使用开源工具将kfb转换为svs

* 基于openslide库将svs转换为jpg图片

## kfb2svs

使用该仓库提供的KFB转SVS软件将kfb转换为svs

[kfb2svs](https://github.com/tcmyxc/kfb2svs)

## svs2jpg

基于openslide库将svs转化为jpg图片

### openslide库安装

window下openslide安装教程：[windows10环境下openslide安装教程](https://blog.csdn.net/Tsehooo/article/details/111249064?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-111249064-blog-125790774.pc_relevant_3mothn_strategy_recovery&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

### 图片转换

使用本仓库的sys2jpg.ipynb代码对svs图像进行批量转换