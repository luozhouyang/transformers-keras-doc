# 安装

`transformers-keras`是使用`Keras`实现的基于`Transformer`模型的库，它可以加载预训练模型的权重，也实现了多个下游的NLP任务。

项目主页：[transformers-keras](https://github.com/luozhouyang/transformers-keras)


## 使用pip安装

transformers-keras可以直接使用pip安装：

```bash
pip install -U transformers-keras
```

使用pip安装的方式会自动把所需要的依赖都安装好。


如果你需要手动安装这些依赖，可以查看下面的依赖列表：

* [tensroflow>=2.0.0](https://github.com/tensorflow/tensorflow)
* [tensorflow-addons](https://github.com/tensorflow/addons)
* [tokenizers](https://github.com/huggingface/tokenizers)
* [seqeval](https://github.com/chakki-works/seqeval)
