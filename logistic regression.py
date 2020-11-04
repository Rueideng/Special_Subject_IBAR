from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import nltk
import numpy as np
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix,classification_report
from wordcloud import WordCloud
import seaborn as sns




#stemming
#cars=>car, ate=>eat, fancier=>fancy, saddest=>sad
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('sentiment_data/stopwords.txt'))
# positive_reviews = BeautifulSoup(open('Train/Pos_train.csv',encoding="utf-8").read(), features="html.parser")
# positive_reviews = positive_reviews.findAll('review_text')
csvfile1 = open('./Train/Pos_train.csv', 'r', encoding='utf-8')
positive_reviews = csv.reader(csvfile1)


# negative_reviews = BeautifulSoup(open('Train/Neg_train.csv',encoding="utf-8").read(), features="html.parser")
# negative_reviews = negative_reviews.findAll('review_text')
csvfile2 = open('./Train/Neg_train.csv', 'r', encoding='utf-8')
negative_reviews  = csv.reader(csvfile2)



#tokenizer, stopwords, length<2
def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


#   創建輸入矩陣
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)   # last element is for the label (1:positive, 0:negative)
    for t in tokens:
        # print(t)
        i = word_index_map[t]   #如果這個單字在word_index_map裡面有，那位置=i
        # print(i)
        x[i] += 1   #建立詞頻
    x = x / x.sum()     # normalize it before setting label(在設置標籤之前將其標準化) 歸一化 有多種方法
    x[-1] = label   # X最後一個是label
    return x



word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

#選擇punkt，wordnet
#punkt是英文標記器
#wordnet是英語corpra

#開始處理postive_reviews文字
for review in positive_reviews:
    if positive_reviews.line_num == 1:
        continue
    else:
        orig_reviews.append(str(review))
        tokens = my_tokenizer(str(review))
        if tokens == []:
            pass
        else:
            positive_tokenized.append(tokens)
            # 建立詞彙庫
            for token in tokens:
                # 如果這單字沒有在詞庫
                if token not in word_index_map:
                    # 那就讓他加入一個位置(假設A的位置是0，B沒有在詞庫，那就加入B並根據下面current_index += 1使A位置0 +1 =1，然後1就會是B的位置)
                    word_index_map[token] = current_index
                    current_index += 1
    #break


#開始處理negative_reviews文字
for review in negative_reviews:
    if negative_reviews.line_num == 1:
        continue
    else:
        orig_reviews.append(str(review))
        tokens = my_tokenizer(str(review))
        if tokens == []:
            pass
        else:
            negative_tokenized.append(tokens)
            # 建立詞彙庫
            for token in tokens:
                #如果這單字沒有在詞庫
                if token not in word_index_map:
                    # 那就讓他加入一個位置(假設A的位置是0，B沒有在詞庫，那就加入B並根據下面current_index += 1使A位置0 +1 =1，然後1就會是B的位置)
                    word_index_map[token] = current_index
                    current_index += 1
    #break


# 詞庫總長度
print("len(word_index_map):", len(word_index_map))

# 總共幾條評論
N = len(positive_tokenized) + len(negative_tokenized)
print("N",N)


#data=N x D+1 matrix 建立矩陣 +1是因為label
data = np.zeros((N, len(word_index_map) + 1))


#處理postive_tokenized到矩陣，最後一列是1
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1


#處理negative_tokenized為矩陣，最後一列是0
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1
    # print(i)



# 整理數據並創建訓練/測試拆分

orig_reviews, data = shuffle(orig_reviews, data)

X = data[:,:-1]
Y = data[:,-1]
print(X)
print(Y)


# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# print(Xtrain.shape)
# print(Ytrain.shape)
# print(Xtest.shape)
# print(Ytest.shape)



#class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
model = LogisticRegression(penalty='l2', max_iter=60,)
model.fit(Xtrain, Ytrain)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))



#===========儲存model
# pkl_filename = 'LogisticRegression_model_to_100.pkl'
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)


# # # =========================================載入模型   load the model from disk
# with open(pkl_filename, 'rb') as file:
#     pickle_model = pickle.load(file)
# # Calculate the accuracy score and predict target values
# score = pickle_model.score(Xtest, Ytest)
# print("Test score: {0:.2f} %".format(100 * score))
# Ypredict = pickle_model.predict(Xtest)






# 每個單字權重分數
threshold = 0.5
all_words =""
data_weight=[]
weight_test=[]
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
        all_words = ' '.join(word)
        weight_test=[word, weight]
        data_weight.append(weight_test)
# df = pd.DataFrame(columns=["單字","分數"])
# dff = df.append(pd.DataFrame(data_weight, columns=["單字","分數"]))
# dff.to_csv(r'./weight_0914.csv', index=False, encoding="utf-8-sig")


# check misclassified examples
predsincsv={}
preds = model.predict(X)
print(preds)
P = model.predict_proba(X)[:,1]



# 打印“最”錯誤的樣本
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None
for i in range(N):
    p = P[i]
    y = Y[i]
    if y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
    elif y == 0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p
print("Most wrong positive review (prob = %s, pred = %s):" % (minP_whenYis1, wrong_positive_prediction))
print("True Negative, the artitle: ", wrong_positive_review)
print("Most wrong negative review (prob = %s, pred = %s):" % (maxP_whenYis0, wrong_negative_prediction))
print("False Postive, the artitle: ", wrong_negative_review)