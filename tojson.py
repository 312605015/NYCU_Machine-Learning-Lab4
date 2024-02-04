# -*- coding: utf-8 -*-
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# 讀取Excel檔案
df = pd.read_excel("C:\\software\\python\\biglanguage\\AI.xlsx")

# 轉換每一行為指定的JSON格式
def convert_row_to_json(row):
    # 指令部分的固定文本
    instruction = "請根據以下輸入回答選擇題，並以數字回答:\n"
    # 將文章、問題及選項內的換行符號替換成JSON中的換行表示，並將選項編入input中
    input_text = (row['文章'].replace('\n', '\n') + "\n\n" + "請問: " + row['問題'].replace('\n', '\n') +
                  "\n1: " + str(row['選項1']).replace('\n', '\\n') +
                  "\n2: " + str(row['選項2']).replace('\n', '\\n') +
                  "\n3: " + str(row['選項3']).replace('\n', '\\n') +
                  "\n4: " + str(row['選項4']).replace('\n', '\\n') + "\n")
    # 正確答案僅包含答案數字
    output = str(row['正確答案'])
    
    # 組合成最終的JSON物件
    json_entry = {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }
    
    return json_entry

# 將函數應用到每一行數據，並將結果保存在一個列表中
json_data = [convert_row_to_json(row) for index, row in df.iterrows()]

train_data, validation_data = train_test_split(json_data, test_size=0.2, random_state=42)

with open('C:\\software\\python\\biglanguage\\train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('C:\\software\\python\\biglanguage\\validation_data.json', 'w', encoding='utf-8') as f:
    json.dump(validation_data, f, ensure_ascii=False, indent=2)
