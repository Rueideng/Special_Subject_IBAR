"""


將所有翻譯後+表情文字化的評論，結合再一起


"""
import csv
from textblob import TextBlob
import pandas as pd
com= ''
tem=[]

list_try=[]

csvfile = open('./all_user_subnull_trans_emoji.csv', 'r',encoding='utf-8',errors='ignore')
# csvfile = open('./finalV2.csv', 'r',encoding='utf-9',errors='ignore')
reader = csv.reader(csvfile)

#存
df = pd.DataFrame(columns=['酒名','全部評論'])
data = {"name": "","all_comment":""}
cocktail_list = []

n=1

for row in reader:
    if reader.line_num == 1:  # 去除第一列(欄位名稱)
        continue	# skip first row
    else:
        name = row[0]
        comment = row[2]
        if n == 1:
            temporarily= name
            # print("酒名: ",name)
            # print("評論:",comment)
            com = com + ' ' + comment
            n=n+1
        else:
            if name == temporarily:
                com = com + ' ' + comment
                # print("酒名: ", name)
                # print("評論:", com)
            elif name != temporarily: #如果目前酒名不等於前一個酒名(代表進入下個酒名)
                # list_try.append(temporarily)
                # list_try.append(com)
                data['name'] = temporarily #把酒名跟
                data['all_comment'] = com
                temporarily=name #讓暫存酒名=現在酒名
                com="" #評論重設
                com = com + ' ' + comment #重新開始加評論
                cocktail_list.append(list(data.values())) #存進list
            if name == "relativity-whiskey":
                data['name'] ="relativity-whiskey"
                data['all_comment'] ="Sweet like bourbon but smooth like Irish whisky."
                cocktail_list.append(list(data.values()))#存進list
    # print(cocktail_list)
dff = df.append(pd.DataFrame(cocktail_list, columns=['酒名','全部評論']))
dff.to_csv(r'./all_comment.csv', index=False, encoding="utf-8-sig")

