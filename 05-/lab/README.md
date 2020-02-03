# 基于GMM-HMM的语音识别系统

## 要求和注意事项
1. 认真读lab2.pdf, 思考lab2.txt中的问题
2. 理解数据文件
3. ref文件作为参考输出，用diff命令检查自己的实现得到的输出和ref是否完全一致
4. 实验中实际用的GMM其实都是单高斯
5. 阅读util.h里面的注释，Graph的注释有如何遍历graph中state上所有的arc的方法。
6. 完成代码
    * lab2_vit.C中一处代码
    * gmm_util.C中两处代码
    * lab2_fb.C中两处代码

## 作业说明

## 安装
该作业依赖g++, boost库和make命令，按如下方式安装：
* MAC: brew install boost (MAC下g++/make已内置）
* Linux(Ubuntu): sudo apt-get install make g++ libboost-all-dev
* Windows: 请自行查阅如何安装作业环境。

### 编译
对以下三个问题，均使用该方法编译。
``` sh
make -C src
```

### p1 
* 内容：完成lab2_vit.C中的用viterbi解码代码.
* 运行:
    * ./lab2_p1a.sh
    * ./lab2_p1b.sh
* 比较结果: 比较你的程序运行结果p1a.chart和参考结果p1a.chart.ref，可以使用vimdiff p1a.chart p1a.chart.ref进行比较，浮点数值差在一定范围内即可。

### p2 
* 内容：估计模型参数,不使用前向后向算法计算统计量，而是用viterbi解码得到的最优的一条序列来计算统计量，叫做viterbi-EM. 给定align（viterbi解码的最优状态(或边）序列)，原始语音和GMM的初始值，更新GMM参数。完成src/gmm_util.C中两处代码。
* 运行：./lab2_p2a.sh
* 比较结果： 如p1，比较p2a.gmm p2a.gmm.ref

### p3 
* 用前向后向算法来估计参数，完成src/lab2_fb.C中的两处代码。
* 运行：
    * ./lab2_p3a.sh: 1条数据，1轮迭代
    * ./lab2_p3b.sh: 22条数据，1轮迭代
    * ./lab2_p3c.sh: 22条数据，20轮迭代
    * ./lab2_p3d.sh: 使用p3c的训练的模型，使用viterbi算法解码，结果应该和p1b的结果一样一样
* 比较结果: 如p1，分别比较p3a_chart.dat/p3a_chart.ref和p3b.gmm/p3b.gmm.ref。
