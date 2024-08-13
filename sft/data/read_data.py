import json

# 读取JSON文件
with open('alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 保留前3000条数据
filtered_data = data[:3000]

# 将结果写回到新的JSON文件中
with open('data.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print(f'保留了前{len(filtered_data)}条数据，并保存为filtered_data.json')
