# 快速入门

本章节将会带领大家快速入门transformers-keras的使用。

transformers-keras功能强大，它可以：

* 加载不同的预训练模型权重
* 使用预训练模型微调下游任务

与其它类似的库相比，transformers-keras有以下优势：

* 清晰简单的API，使用纯粹的keras构建模型，没有任何多余的封装
* 常用任务的数据管道构建，准备好数据就可以开始训练模型，不需要担心数据处理逻辑
* 可以直接导出SavedModel格式的模型，使用tensorflow/serving直接部署
* 最小依赖，除了tensorflow，不附带任何其它庞大的第三方库

## 加载预训练模型的权重

## 使用预训练模型微调下游任务