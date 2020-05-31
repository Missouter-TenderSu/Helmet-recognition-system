# 慧识-智能安全帽检测系统

第十一届中国大学生服务外包创新创业大赛 【A05】基于人工智能的视觉识别技术【文思海辉】解决方案

## 推荐运行环境

#### 硬件环境

- CPU：Intel i5-8300H 或更佳
- 显卡：NVDIA  GeForce GTX 1060 或更佳
- 内存：DDR4 8GB 或更大
- 硬盘：80GB 或更大

#### 软件环境
- Ubuntu 18.04  LTS
- NVDIA 显卡驱动程序
- PyTorch 1.4.1
- CUDA 10.2
- NCCL 2
- GCC 7.4.0
- mmcv
- mmdetection

## 文件结构
    /mmdet					注意:
    						环境配置完毕后,将 /mmdet 、 /configs 
    /configs				与mmdetectron下的同名文件夹合并,并替换同名文件。
    
    /work_dirs 				存放训练完毕的算法模型,在调用时需自行设置路径。
    
    /icon 					存放GUI的素材文件
    
    APowefulCamera.py		GUI逻辑功能文件
    
    APowefulCamera.ui		QtDeisgner设计文件
    
    Ui_APowefulCamera.py	GUI界面文件
    
    image.py				
    注意！image.py需要替换anaconda3/envs/(your env name)/lib/python3.x/site-packages/mmcv/visualization/image.py


## 运行指令

本系统运行指令如下:

    python "APowerfulCamera.py"



## 声明

本项目库专门存放 第十一届服创大赛 A05 相关代码文件

如有任何对代码的问题请邮箱联系：[missouter@yandex.com](mailto:missouter@yandex.com)

