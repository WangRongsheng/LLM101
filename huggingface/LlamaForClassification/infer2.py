from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from peft import PeftModel

mode_path = 'roberta-large'
lora_path = 'save_checkpoints_roberta/checkpoint-2380'

print("load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

print("load model")
model = AutoModelForSequenceClassification.from_pretrained(mode_path, device_map="auto", offload_folder="offload", trust_remote_code=True).eval()

print("load lora model")
model = PeftModel.from_pretrained(model, model_id=lora_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
print("load finish")

inputs = tokenizer(text="what time does your talk go until? I don't know if I can make it due to work.", truncation=True, padding=True, return_tensors="pt")
inputs = inputs.to(model.device)

output = model(**inputs)
prediction = output.logits.argmax(dim=-1).item()
#print(output.logits)

min_val = output.logits.min()
max_val = output.logits.max()
normalized_tensor = (output.logits - min_val) / (max_val - min_val)
normalized_tensor = normalized_tensor.cpu().detach().tolist()[0]
#print(max(normalized_tensor))

#print(prediction)

print(f"Prediction Label:{prediction}, Confidence:{max(normalized_tensor)}")
