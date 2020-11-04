"""
利用word2vec擴增單字庫

"""


import csv
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from gensim.models import word2vec
weight_list=[]


def get_two_float(f_str, n):
    f_str = str(f_str)      # f_str = '{}'.format(f_str) 也可以轉換為字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 傳入的函數有幾位小數，在字符串後面都添加n為小數0
    return ".".join([a, c])

def merge_two_dicts(x, y):
# """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def load_model(words):# 載入模型
    model = word2vec.Word2Vec.load("./word2vec.model")
    # print(model.wv.similar_by_word(words)) #顯示輸入的關鍵字的同義字
    # for co in (model.wv.similar_by_word(words)[:5]): #取出同義字
    return (model.wv.similar_by_word(words)[:5])



dict_data={"word":"","score":""}
dictdata={}
dictdata_add={}
dictupdate={}
num=0

# 開啟 CSV 檔案
with open('whisky_weight_word2vec_add.csv', newline='',encoding='utf8') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    # 以迴圈輸出每一列
    for row in rows:
        if rows.line_num == 1:
            continue
        else:
            dictdata[row[0]]=row[1] #將單字(key) 與 分數(value) 存到dictdata字典
            num+=1


print("詞庫目前的單字詞: ",num)



for word in dictdata:
    value=dictdata[word] #字典中的值=value
    try:
        word2v=load_model(word) #把單字送進去word2vec 取出前五個同義字
    except:
        continue

    # print(word2v)
    for co in word2v: #從這五個同義字一個一個取出
       dictdata_add[co[0]] = value #設定新的字典 同義字加進去此字典 並把所有同義字的分數 = word的分數
merge = merge_two_dicts(dictdata_add,dictdata) # 兩個字典合併
expansion = len(merge)
# print(merge)
print("擴增後的單字數量: ",expansion)
print("共增加: ",expansion-num)
#存進去CSV
weight_df = pd.DataFrame(columns=['單字','權重'])
for k in merge:
    dict_data["word"]=k
    dict_data["score"]=merge[k]
    weight_list.append(list(dict_data.values()))
# print(weight_list)
weight_dff = weight_df.append(pd.DataFrame(weight_list, columns=['單字','權重']))
weight_dff.to_csv(r'./whisky_weight_word2vec_add_again.csv', index=False, encoding="utf-8-sig")
