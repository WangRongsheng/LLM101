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
    
