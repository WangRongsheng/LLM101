from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from peft import PeftModel

#mode_path = 'NousResearch/Llama-2-7b-hf'
mode_path = 'roberta-large'
#lora_path = 'save_checkpoints_llama/checkpoint-2380' # 这里改称你的 lora 输出对应 checkpoint 地址
lora_path = 'save_checkpoints_roberta/checkpoint-2380' # 这里改称你的 lora 输出对应 checkpoint 地址

print("load tokenizer")
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
# 加载模型
print("load model")
model = AutoModelForSequenceClassification.from_pretrained(mode_path, device_map="auto", offload_folder="offload", trust_remote_code=True).eval() # device_map="auto" ,torch_dtype=torch.bfloat16
# 加载LoRA权重
print("load lora model")
model = PeftModel.from_pretrained(model, model_id=lora_path)
print("load finish")

inputs = tokenizer(text="Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all", truncation=True, padding=True, return_tensors="pt")
output = model(**inputs)

prediction = output.logits.argmax(dim=-1).item()
print(prediction)

#print(f'\n Class: {prediction}, Label: {id2label[prediction]}, Text: {text}')
