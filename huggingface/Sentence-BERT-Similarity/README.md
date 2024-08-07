BERT和RoBERTa在文本语义相似度等句子对的回归任务上，已经达到了SOTA的结果。但是，它们都需要把两个句子同时喂到网络中，这样会导致巨大的计算开销。这种结构使得BERT不适合语义相似度搜索，同样也不适合无监督任务（例如：聚类）。Sentence-BERT(SBERT)网络利用孪生网络和三胞胎网络结构生成具有语义意义的句子embedding向量，语义相近的句子其embedding向量距离就比较近，从而可以用来进行相似度计算(余弦相似度、曼哈顿距离、欧式距离)。这样SBERT可以完成某些新的特定任务，例如相似度对比、聚类、基于语义的信息检索。

### 1. 下载数据集和预训练模型

- 数据：https://opendatalab.com/OpenDataLab/LCQMC
- 预训练模型：https://huggingface.co/hfl/chinese-roberta-wwm-ext

```
总共有数据：260068条，其中正样本：149226条，负样本：110842条
训练数据：234061条,其中正样本：134330条，负样本：99731条
测试数据：26007条,其中正样本：14896条，负样本：11111条
```

### 2. 训练

```python
CUDA_VISIBLE_DEVICES=4 python train.py
```

### 3. 评估

```python
CUDA_VISIBLE_DEVICES=4 python evaluate.py
```

```
				precision    recall  f1-score   support
           0       0.93      0.83      0.88     11111
           1       0.89      0.95      0.92     14896
    accuracy                           0.90     26007
   macro avg       0.91      0.89      0.90     26007
weighted avg       0.90      0.90      0.90     26007
```

### 4. 预测

```python
import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util

model_path = 'model/save_model/'
model = SentenceTransformer(model_path)

s1 = "开初婚未育证明怎么弄？"
s2 = "初婚未育情况证明怎么开？"

embedding1 = model.encode(s1, convert_to_tensor=True)
embedding2 = model.encode(s2, convert_to_tensor=True)

similarity = util.cos_sim(embedding1, embedding2).item()
print(f"句子1：{s1}\n句子2：{s2}\n句子1和句子2的相似度是{similarity}")

#句子1：开初婚未育证明怎么弄？
#句子2：初婚未育情况证明怎么开？
#句子1和句子2的相似度是0.9386134147644043
```

### 引用

如果你觉得我们的工作对你有用，欢迎引用它。

```bibtex
@Misc{wang2024sbert,
  title = {Sentence-BERT-Similarity},
  author = {WangRongsheng},
  howpublished = {\url{https://github.com/WangRongsheng/Sentence-BERT-Similarity}},
  year = {2024}
}
```