# 下游任务模型概述


## 统一的API


## Question Answering模型介绍


## Sentence Embedding模型介绍


## Sentiment Analysis模型介绍

细粒度的情感分析，即`Aspect-based Sentiment Analysis`，通常会把它作为一个**序列标注任务**。也就是说，会对一个序列的 **Aspect Term** 和 **Opinion Term**进行序列标注，例如使用常见的**B-I-O**标注。标注过程中，可以同时对Aspect进行分类，对Opinion进行情感极性分类。

这种方式有一个缺点就是，对于类别经常变化的情况不太友好，一般来说需要重新训练模型，或者干脆对每一个领域训练一个单独的序列标注模型。总之，这种方式还是显得比较麻烦。

我本人更喜欢以下的方式：把Aspect Term Extraction、Opinion Term Extraction、Opinion Sentiment Classification三个任务，都使用Question Answering的方式来处理。

也就是说，Aspect-based Sentiment Analysis可以分成两个部分：
* Aspect Term Extraction，只需要抽出Aspect Term，不需要进行分类
* Opinion Term Extraction & Classification，需要抽取出Opinion Term，同时进行情感极性分类

这两个部分使用两个独立的模型来处理，但是都是使用Question Answering的方式基于BERT模型实现。


### Aspect Term Extraction


### Opinion Term Extraction and Classification


## Sequence Classification模型介绍


## Token Classification模型介绍
