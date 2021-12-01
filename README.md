# Transformer-ASR使用说明

## 1.简要说明

目前支持功能：对LibriSpeech ASR数据集的运行脚本； 通过读取yaml配置文件进行训练； 速度扰动、SpecAugmet； Conformer结构模型  
详细内容见报告：2101798-王杰
<br/>
<br/>
## 2.安装

* 本次实验用到的环境

1. Python == 3.8.10
2. torch == 1.7.1, torchaudio == 0.7.0, cuda == 10.2
3. apex
```
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
4. nccl
```
make -j src.build CUDA_HOME=<path to cuda install>
```
5. gcc 5.4
6. python library 
```
pip install pandas sentencepiece configargparse gpustat tensorboard editdistance
```
<br/>
<br/>

## 3.代码结构
以librispeech/asr为例：
```markdown
librispeech  
├── conf  
├── local  
│   ├── monitor.sh  
│   ├── parse_options.sh  
│   ├── path.sh  
│   └── utils.sh  
├── decode.sh  
├── run.sh  
└── train.sh  
```

- run.sh是核心脚本，包含了数据的处理以及模型的训练及解码，train.sh和decode.sh分别调用run.sh来实现单独的训练和解码功能。
- conf文件夹下为训练配置，支持读取yaml文件。模型训练所要使用的配置可以在该文件中进行设置。
- local文件夹下为一些可以选用的常用脚本

<br/>
<br/>
  
## 3.模型测试结果部分ASR实例展示
Generate test-clean with beam=5: WER: 6.44&emsp;&emsp;Generate test-other with beam=5: WER: 12.60  
模型在LibriSpeech数据集的test-clean测试集上上的识别准确率达到了93.56%
<br/>
<br/>



## 4.部分ASR实例展示
|参考文本|识别文本|
|---|---|
|this is our last feast with you i said	|this is our last feast with you i said|
|we have a commander who's game for anything|we have a commander whose game for anything|
|unlucky me and the mother that bore me	|and lucky me into the mother that bore me|
|but now nothing could hold me back|	but now nothing could hold me back|
|he never loses sight of the purpose of his epistle	|he never loses sight of the purpose of his epistle|
|oh but i'm glad to get this place mowed	|oh but i'm glad to get this place mode|
|it's tremendously well put on too	|it's tremendously well put on too|
|we never had so many of them in here before	|we never had so many of them in here before|
|there are few changes in the old quarter|	there are few changes in the old quarter|
|i didn't preach without direction|	i didn't preach without direction|  

全部的识别结果在results文件夹里

<br/>
<br/>

## 5.训练完成的模型下载地址
模型下载地址：链接：https://pan.baidu.com/s/1Ft0lfk5Y6LYxuDsgSat4_A 提取码：1234   
需要数据集的可以联系邮箱wangcoder@outlook.com


