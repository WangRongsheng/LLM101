from torch.utils.data import  DataLoader
import torch.nn as nn
from sentence_transformers import  SentenceTransformer, InputExample, losses
from sentence_transformers import models, evaluation
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(data_file):
    data = pd.read_csv(data_file, sep='\t', header=None, names=['index', 's1', 's2', 'label'])

    x = data[['s1', 's2']].values.tolist()
    y = data['label'].values.tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1, shuffle=True)
    print(x_train[0], y_train[0])
    print('总共有数据：{}条，其中正样本：{}条，负样本：{}条'.format(
        len(x), sum(y), len(x) - sum(y)))
    print('训练数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_train), sum(y_train), len(x_train) - sum(y_train)))
    print('测试数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_test), sum(y_test), len(x_test) - sum(y_test)))
    return x_train, x_test, y_train, y_test

model_path = 'model/chinese-roberta-wwm-ext/'
word_embedding_model = models.Transformer(model_path, max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=256, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

x_train, x_test, y_train, y_test = get_data(data_file = 'data/data.csv')
train_examples = []
for s, label in zip(x_train, y_train):
    s1, s2 = s
    train_examples.append(
        InputExample(texts=[s1, s2], label=float(label))
    )
test_examples = []
for s, label in zip(x_test, y_test):
    s1, s2 = s
    test_examples.append(
        InputExample(texts=[s1, s2], label=float(label))
    )
train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model_save_path = 'model/save_model/'
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
model.fit(train_objectives=[(train_loader, train_loss)],
          epochs=5,
          evaluator=evaluator,
          warmup_steps=100,
          save_best_model=True,
          output_path=model_save_path,)