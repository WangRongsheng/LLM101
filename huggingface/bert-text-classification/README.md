# bert_text_classification

## 数据集

从 THUCNews 中随机抽取20万条新闻标题，一共有10个类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐，每类2万条标题数据。数据集按如下划分：

- 训练集：18万条新闻标题，每个类别的标题数为18000
- 验证集：1万条新闻标题，每个类别的标题数为1000
- 测试集：1万条新闻标题，每个类别的标题数为1000

可以按照 data 文件夹中的数据格式来准备自己任务所需的数据，并调整 config.py 中的相关配置参数。

## 预训练 BERT 模型

从 huggingface 官网上下载 bert-base-chinese 模型权重、配置文件和词典到 pretrained_bert 文件夹中，下载地址：https://huggingface.co/bert-base-chinese/tree/main

## 模型训练

文本分类模型训练：

```shell
python main.py --mode train --data_dir ./data --pretrained_bert_dir ./pretrained_bert
```

### Demo演示

文本分类 demo 展示：

```shell
python main.py --mode demo --data_dir ./data --pretrained_bert_dir ./pretrained_bert
```

### 模型预测

对 data 文件夹下的 input.txt 中的文本进行分类预测：

```shell
python main.py --mode predict --data_dir ./data --pretrained_bert_dir ./pretrained_bert --input_file ./data/input.txt
```


