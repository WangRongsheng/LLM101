import json
import csv
import pandas as pd

def trans_json2csv(input_file, output_path):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]

    for item in data:
        item.pop("ID", None)

    csv_file_name = output_path

    with open(csv_file_name, 'w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')

        i = 0
        for item in data:
            i = i+1
            row_data = [i, item["sentence1"], item["sentence2"], item["gold_label"]]
            csv_writer.writerow(row_data)

if __name__ == '__main__':
    # 单文件csv转化
    #trans_json2csv("LCQMC_train.json", "train.csv")
    #trans_json2csv("LCQMC_dev.json", "dev.csv")
    #trans_json2csv("LCQMC_test.json", "test.csv")
    
    # 合并json转化为csv
    trans_json2csv("data.json", "data.csv")