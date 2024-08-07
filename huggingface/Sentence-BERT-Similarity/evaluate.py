import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(data_file):
    data = pd.read_csv(data_file, sep='\t', header=None, names=['index', 's1', 's2', 'label'])

    x = data[['s1', 's2']].values.tolist()
    y = data['label'].values.tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123, shuffle=True)
    print(x_train[0], y_train[0])
    print('总共有数据：{}条，其中正样本：{}条，负样本：{}条'.format(
        len(x), sum(y), len(x) - sum(y)))
    print('训练数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_train), sum(y_train), len(x_train) - sum(y_train)))
    print('测试数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_test), sum(y_test), len(x_test) - sum(y_test)))
    return x_train, x_test, y_train, y_test

model_path = 'model/save_model/'
model = SentenceTransformer(model_path)

x_train, x_test, y_train, y_test = get_data(data_file = 'data/data.csv')

s1 = np.array(x_test)[:, 0]
s2 = np.array(x_test)[:, 1]
embedding1 = model.encode(s1, convert_to_tensor=True)
embedding2 = model.encode(s2, convert_to_tensor=True)

pre_labels = [0] * len(s1)
predict_file = open('predict.txt', 'w')
for i in range(len(s1)):
    similarity = util.cos_sim(embedding1[i], embedding2[i])
    if similarity > 0.5:
        pre_labels[i] = 1
    predict_file.write(s1[i] + ' ' +
                       s2[i] + ' ' +
                       str(y_test[i]) + ' ' +
                       str(pre_labels[i]) + '\n')
print(classification_report(y_test, pre_labels))
predict_file.close()
