#  ChatGirl 一个基于 TensorFlow Seq2Seq 模型的聊天机器人[中文文档]

## 简介
简单地说就是该有的都有了，但是总体跑起来效果还不好。
还在开发中，它工作的效果还不好。但是你可以直接训练，并且运行。
包含预处理过的 twitter 英文数据集，训练，运行，工具代码，可以运行但是效果有待提高。


## 数据集

Twitter 数据集:

> https://github.com/suriyadeepan/datasets

## 训练
你需要新建一个  model 文件夹来保存训练完的模型
运行这个文件来训练模型

Train_Model.py

## 运行
运行这个文件来运行模型

Run_model.py

## 工具集

w2idx:单词到 id ，用于下一步将单词转化为向量。

idx2w:id 到 单词，用于将 id 转换为单词。

hello.py 中有两个函数的 demo。


## 结果

Result.md(部分输出)

## 博客

http://www.tensorflownews.com/

## 开发计划
* 优化数据集
* 优化模型

[under developing]


## TensorFlowNews 项目推荐：

基于 TensorFlow 的神经网络三大主流模型卷积神经网络，循环神经网络，序列到序列模型 TensorFlowNews 都原创了实战项目，包含模型，代码，数据集，工具集，欢迎 star ：

* TensorFlow CNN Model Project:

FaceRank - Rank Face by CNN Model based on TensorFlow (add keras version). FaceRank-人脸打分基于 TensorFlow (新增 Keras 版本) 的 CNN 模型（可能是最有趣的 TensorFlow 中文入门实战项目）

[https://github.com/fendouai/FaceRank](https://github.com/fendouai/FaceRank)

* TensorFlow LSTM Model Project:

一个比特币交易机器人基于 Tensorflow LSTM 模型，仅供娱乐。 A Bitcoin trade robot based on Tensorflow LSTM model.Just for fun.

[https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot](https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot)

* TensorFlow Seq2Seq Model Project:

ChatGirl is an AI ChatBot based on TensorFlow Seq2Seq Model.ChatGirl 一个基于 TensorFlow Seq2Seq 模型的聊天机器人。（包含预处理过的 twitter 英文数据集，训练，运行，工具代码，可以运行但是效果有待提高。）

[https://github.com/fendouai/ChatGirl](https://github.com/fendouai/ChatGirl)


## 学习流程
如果看到这里有很多不懂的话，建议：
*  Hello World
https://zhuanlan.zhihu.com/p/27963600
*  基本概念
https://zhuanlan.zhihu.com/p/27986689
*  卷积神经网络
https://zhuanlan.zhihu.com/p/28161292
*  训练好模型参数的保存和恢复代码
https://zhuanlan.zhihu.com/p/27912379
* TensorFlowNews 专栏
https://zhuanlan.zhihu.com/TensorFlownews
*  TensorFlowNews 博客
http://www.tensorflownews.com/

欢迎关注我的博客，因为我也还在学习中，现有的教程经常比较大，涉及到的只是比较多，我会经常拆分出小的知识点，我的博客也会把这些小的知识点记录下来。
