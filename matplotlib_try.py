import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame,Series


"""#====================================
將羅吉斯回歸產生單字分數進行排名(排序)
將酒進行排名(排序)

最後產生前五名圖片
""""====================================
dictdata={"total":""}
data={}

slist_Whiskey=[]
slist_Whiskey_name=[]
slist_Whiskey_analysis=[]
slist_Whiskey_weight=[]

xlist_Whiskey=[]
xlist_Whiskey_name=[]
xlist_Whiskey_analysis=[]
xlist_Whiskey_weight=[]


name=[]

slist=[]
slist_name=[]
slist_weight=[]

xlist=[]
xlist_name=[]
xlist_weight=[]

total_whiskey=[]
dict_whiskey={"Name":"","Score":"","Rank":""}

def get_two_float(f_str, n):
    f_str = str(f_str)     # f_str = '{}'.format(f_str) 也可以轉換為字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 傳入的函數有幾位小數，在字符串後面都添加n為小數0
    return ".".join([a, c])


class ListDict(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age


class ListDict_Whiskey(object):
    def __init__(self, name, analysis, score):
        self.name = name
        self.analysis = analysis
        self.score = score


#載入單字權重分數，進行單字分數排序
with open('sentiment_whisky_final.csv', newline='',encoding='utf8') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    # 以迴圈輸出每一列
    for row in rows:
        if rows.line_num == 1:
            continue
        else:
            rorw2 = get_two_float(row[2], 2).replace("-0.0","0.0")  # 把分數取小數點
            if float(rorw2) >= 0:
                slist_Whiskey.append(ListDict_Whiskey(row[0], row[1],rorw2))
            else:
                xlist_Whiskey.append(ListDict_Whiskey(row[0], row[1],rorw2))

#載入酒分數，進行酒分數排序
with open('whisky_weight.csv', newline='',encoding='utf8') as csvfile:
    # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            if rows.line_num == 1:
                continue
            else:
                rorw2 = get_two_float(row[1], 1) #把分數取小數點
                if float(row[1]) >= 0 :
                    slist.append(ListDict(row[0],rorw2))
                else :
                    xlist.append(ListDict(row[0], rorw2))

# print("排序前")
# for i in slist:
#     print(i.name,i.age)
# print("按照名字排序")
# slist.sort(key=lambda  x: x.name)
# for i in slist:
#     print(i.name,i.age)
#=========================酒全部排名


rank_df = pd.DataFrame(columns=['酒名','分數','排名'])
num=0
slist_Whiskey.sort(key=lambda  x: x.score,reverse=True)
for i in slist_Whiskey:
    num+=1
    dict_whiskey["Name"] = i.name
    dict_whiskey["Score"] = i.score
    dict_whiskey["Rank"] = num
    total_whiskey.append(list(dict_whiskey.values()))

xlist_Whiskey.sort(key=lambda  x: x.score)
for i in xlist_Whiskey:
    num += 1
    dict_whiskey["Name"] = i.name
    dict_whiskey["Score"] = i.score
    dict_whiskey["Rank"] = num
    total_whiskey.append(list(dict_whiskey.values()))
rank_dff = rank_df.append(pd.DataFrame(total_whiskey, columns=['酒名','分數','排名']))
rank_dff.to_csv(r'./whiskey_rank_sort.csv', index=False, encoding="utf-8-sig")



#==========================酒正負面排名
print("酒-正數按照分數排序")
slist_Whiskey.sort(key=lambda  x: x.score)
for i in slist_Whiskey:
    # print(i.name,i.age)
    slist_Whiskey_name.append(i.name)
    slist_Whiskey_analysis.append(i.analysis)
    slist_Whiskey_weight.append(i.score)
    dict_whiskey[i.name] = i.score

x_Whiskey_1 = slist_Whiskey_name[-5:]
y_Whiskey_1 = slist_Whiskey_weight[-5:]

print(x_Whiskey_1)
print(y_Whiskey_1)


plt.figure(figsize=(10,10),dpi=400,linewidth = 0.25)
plt.subplots_adjust(left=0.4,right=0.9,bottom=0.07,top=0.9,wspace=0.1,hspace=0.1)
plt.barh(x_Whiskey_1,y_Whiskey_1, align =  'center',color=['coral'],edgecolor='red')
plt.title("Whiskey-positive numbers sorted by score")
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
plt.xlabel("Score")
plt.ylabel("Whiskey name")
plt.savefig('Whiskey_positive_numbers_sorted_by_score.png')
plt.show()



print("酒-負數按照分數排序")
xlist_Whiskey.sort(key=lambda  x: x.score)
for i in xlist_Whiskey:
    # print(i.name,i.age)
    xlist_Whiskey_name.append(i.name)
    xlist_Whiskey_analysis.append(i.analysis)
    xlist_Whiskey_weight.append(i.score)
    dict_whiskey[i.name] = i.score

x_Whiskey_2= xlist_Whiskey_name[-5:]
y_Whiskey_2= xlist_Whiskey_weight[-5:]

print(x_Whiskey_2)
print(y_Whiskey_2)

plt.figure(figsize=(10,10),dpi=400,linewidth = 0.25)
plt.subplots_adjust(left=0.4,right=0.9,bottom=0.07,top=0.9,wspace=0.1,hspace=0.1)
plt.barh(x_Whiskey_2, y_Whiskey_2, align =  'center',color="green")
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
plt.title("Whiskey-negative numbers sorted by score")
plt.xlabel("Score")
plt.ylabel("Whiskey name")
plt.savefig('Whiskey_negative_numbers_sorted_by_score.png')
plt.show



#=================================================詞彙正負面排名

print("單字-正數按照分數排序")
slist.sort(key=lambda  x: x.age)
for i in slist:
    # print(i.name,i.age)
    slist_name.append(i.name)
    slist_weight.append(i.age)

x1=slist_name[-5:]
y1=slist_weight[-5:]
print(x1)
print(y1)
# plt.bar(x1, y1, align =  'center')
# plt.title("Matplotlib demo")
# plt.xlabel("x1 axis caption")
# plt.ylabel("y1 axis caption")
# plt.show()




print("負數按照分數排序")
xlist.sort(key=lambda  x: x.age,reverse=True)
for i in xlist:
    # print(i.name,i.age)
    xlist_name.append(i.name)
    xlist_weight.append(i.age)
x2 = xlist_name[:5]
y2 = xlist_weight[:5]
print(x2)
print(y2)
# plt.bar(x2, y2, align =  'center')
# plt.title("Matplotlib demo")
#
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.show


#================================劃出正負面 前10名的單字與分數
yum=[]
x1.extend(x2)
y1.extend(y2)
for i in y1:
    yum.append(float(i))
# #創建畫板獲取axes對象
# fig,axes = plt.subplots(2,1,figsize=(10,10))
#創建畫板獲取axes對象
fig = plt.subplots(1,1,figsize=(10,10))
#調整畫板大小
# fig.set_size_inches(30,10)
# 創建繪圖數據
data = Series(np.array(yum),index = np.array((x1)))


# # 利用series數據在2行1列的畫板上的一塊區域，繪製一個柱狀圖
# data.plot(kind='bar',ax=axes[0],color='RED',alpha=0.7,title="Top 10 positive and negative numbers")
# 利用Series数据在两行一列的画板上的第二块区域绘制柱条形图
data.plot(kind='barh',color='indigo',alpha=0.7,title="Top 5 positive and negative numbers")
plt.savefig('Top_10_positive_and_negative_numbers.png')
plt.show()




