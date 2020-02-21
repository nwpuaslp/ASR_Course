# 第七章作业
# 说明
请根据"实验指导书.pdf":part2.2和part4分别完成N-gram计数和Witten-Bell算法的编写。

编译文件：Makefile

# 提供的C++文件介绍：
1. main.C：入口函数
2. util.{H,C}:提供命令行解析，读取和写出数据等功能,不必仔细阅读，可以掠过。
3. lang_model.{H,C}:LM类定义，本实验主要部分内容，需要完成.C文件中count_sentence_ngrams()和get_prob_witten_bell()函数。
4. lab3_lm.{H,C}:语言模型实验的wrapper函数。

# 数据文件：
	字典：lab3.syms
	训练集：minitrain.txt和minitrain2.txt
	测试集：test1.txt和test2.txt

# bash文件：
	lab3_p1{a,b}.sh:测试N-gram计数
	lab3_p3	{a,b}.sh：测试Witten-Bell smoothing算法
