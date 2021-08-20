# 快速入门

本章节将会带领大家快速入门`transformers-keras`的使用。

`transformers-keras`功能强大，它可以：

* 加载不同的预训练模型权重
* 使用预训练模型微调下游任务

与其它类似的库相比，`transformers-keras`有以下优势：

* 清晰简单的API，使用纯粹的`keras`构建模型，没有任何多余的封装
* 常用任务的数据管道构建，准备好数据就可以开始训练模型，不需要担心数据处理逻辑
* 可以直接导出`SavedModel`格式的模型，使用 [tensorflow/serving](https://github.com/tensorflow/serving) 直接部署
* 最小依赖，除了`tensorflow`，不附带任何其它庞大的第三方库

## 加载预训练模型的权重

基于 **BERT** 的模型支持加载以下预训练权重：

* 所有使用 [google-research/bert](https://github.com/google-research/bert) 训练的 **BERT** 模型
* 所有使用 [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 训练的 **BERT** 和 **RoBERTa** 模型

基于 **ALBERT** 的模型支持加载以下预训练权重:

* 所有使用 [google-research/albert](https://github.com/google-research/albert) 训练的 **ALBERT** 模型


这里是基于 **BERT** 的模型的使用示例。所有基于 **ALBERT** 的模型用法和BERT类似，所以这里只使用 **BERT** 为例，不再重复用ALBERT举例。

### BERT特征抽取示例

我们这里直接加载预训练的BERT权重，来做句子特征的抽取。

```python
from transformers_keras import Bert

# 加载预训练模型权重
model = Bert.from_pretrained('/path/to/pretrained/bert/model')
input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
sequence_output, pooled_output = model(inputs=[input_ids, segmet_ids, attention_mask], training=False)

```

通过上述代码，就可以抽取出模型的`sequence_output`和`pooled_output`特征向量。

其中:

* `sequence_output` 是BERT模型最后的输出状态，是一个形状为 `(batch_size, sequence_length, hidden_size)` 的张量。
* `pooled_output` 是BERT的`[CLS]`位置的向量经过Dense层得到的`pooling`输出。它是一个形状为`(batch_size, hidden_size)` 的张量。


你可以通过这两个输出，采取不同的手段，来获取输入**句子**的向量。例如：

* 使用`mean-pooling`策略计算句子向量
* 使用`[CLS]`策略来获取句子向量( `sequence_output[:, 0, )]` 即为 `[CLS]`的向量表示)
* 使用`pooled_output`直接作为句子的向量



另外，可以通过构造器参数 `return_states=True` 和 `return_attention_weights=True` 来获取每一层的 `hidden_states` 和 `attention_weights` 输出:

```python
from transformers_keras import Bert

# 加载预训练模型权重
model = Bert.from_pretrained(
    '/path/to/pretrained/bert/model', 
    return_states=True, 
    return_attention_weights=True)
input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
sequence_outputs, pooled_output, hidden_states, attn_weights = model(inputs=[input_ids, segment_ids, attention_mask], training=False)

```

其中:

* `hidden_states`就是每一层 `hidden_state` stack在一起的输出。它是一个形状为 `(batch_size, num_layers, sequence_length, hiddeb_size)` 的张量。
* `attn_weights`就是每一层`attention_weights` stack在一起的输出。它是一个形状为 `(batch_size, num_layers, num_attention_heads, sequence_length, sequence_length)` 的张量。

## 使用预训练模型微调下游任务

这里有以下几个示例：

* 使用BERT微调 **文本分类** 任务
* 使用BERT微调 **问答** 任务

### 使用BERT微调文本分类任务

你可以使用BERT构建序列的二分类网络：

```python
from transformers_keras import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('/path/to/pretrained/model')
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
```

可以得到下面的模型输出：
```bash
Model: "bert_for_sequence_classification"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BertModel)                ((None, None, 768),  59740416    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            1538        bert[0][1]                       
==================================================================================================
Total params: 59,741,954
Trainable params: 59,741,954
Non-trainable params: 0
__________________________________________________________________________________________________
```

要训练网络，你需要准备训练数据。训练数据的格式采用`JSONL`格式，即文件的每一行都是一个`JSON`。例如：

```bash
{"sequence": "我喜欢自然语言处理(NLP)", "label": 1}
{"sequence": "我不喜欢自然语言处理(NLP)", "label": 0}
```
需要注意的是，每个**JSON**都需要包含两个字段：

* `sequence`，即文本序列
* `label`，即文本的类别ID

然后，就可以开始构造数据集，训练模型了：

```python
from transformers_keras import SequenceClassificationDataset

input_files = [
    "filea.jsonl",
    "fileb.jsonl"
]

dataset = SequenceClassificationDataset.from_jsonl_files(
    input_files=input_files,
    batch_size=32,
)
# 你可以查看dataset长什么样
print(next(iter(dataset)))

model.fit(
    dataset,
    epochs=10,
)

```

### 使用BERT微调问答任务


另一个例子，使用BERT来做Question Answering：

```python
from transformers_keras import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('/path/to/pretrained/model')
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

```

可以得到下面的模型输出：
```bash
Model: "bert_for_question_answering"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BertModel)                ((None, None, 768),  59740416    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 2)      1538        bert[0][0]                       
__________________________________________________________________________________________________
head (Lambda)                   (None, None)         0           dense[0][0]                      
__________________________________________________________________________________________________
tail (Lambda)                   (None, None)         0           dense[0][0]                      
==================================================================================================
Total params: 59,741,954
Trainable params: 59,741,954
Non-trainable params: 0
__________________________________________________________________________________________________
```

同样的，训练数据使用`JSONL`格式。每一行都是一个JSON，例如：

```bash
{"question": "距离地球最近的天体是", "context": "月亮,你非要选的话,选c好了", "answer": "月亮"}
{"question": "从月球上看地球的唯一建筑物是", "context": "从月球上看地球的唯一建筑物是:中国的万里长城", "answer": "万里长城"}
```

其中，每个JSON需要包含以下字段：

* `context`，即上下文
* `question`，即问题
* `answer`，即答案

准备好数据集，就可以开始训练了：

```python
from transformers_keras import QuestionAnsweringDataset

input_files = ["filea.jsonl", "fileb.jsonl"]

dataset = QuestionAnsweringDataset.from_jsonl_files(
    input_files=input_files,
    batch_size=32,
)

# 你可以查看dataset长什么样
print(next(iter(dataset)))

model.fit(
    dataset,
    epochs=10,
)
```


`transformers-keras`还支持其它不同的任务：

* 序列标注，例如NER、POS
* 句向量，例如SimCSE

这些任务会在下面的文档里介绍～。