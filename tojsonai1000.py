import pandas as pd
import json

# 讀取Excel文件
df = pd.read_excel("C:\\software\\python\\biglanguage\\AI1000.xlsx")

# 傳換特定行數為JSON格式
def convert_row_to_json(row):
    # 指令部分的固定文本
    instruction = "請根據以下輸入回答選擇題，並以數字回答:"
    
    
    input_text = (row['文章'] + "\n\n" + "請問: " + row['問題'] +
                  "\n1: " + str(row['選項1']) +
                  "\n2: " + str(row['選項2']) +
                  "\n3: " + str(row['選項3']) +
                  "\n4: " + str(row['選項4']) + "\n")
    
    
    json_entry = {
        "id": row['題號'],
        "instruction": instruction,
        "input": input_text
    }
    return json_entry


json_data = [convert_row_to_json(row) for index, row in df.iterrows()]

with open('C:\\software\\python\\biglanguage\\AI1000.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)
