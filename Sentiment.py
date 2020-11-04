"""
#執行把現有評論(酒有多條評論整合成一款酒只有一條)生成分數
#執行把現有評論(酒有多條評論)一條一條生成分數
#自行輸入評論生成分數
#重複輸入評論生成分數

"""
import csv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import pandas as pd
import re
from gensim.models import word2vec"
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('../路徑/stopwords.txt'))


weight_list=[]
minmax_score={}
#============================================================================================================
def get_wordnet_pos(tag): #詞性還原
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#============================================================================================================

def part_of_speech_reduction(sentence):
    tokens = word_tokenize(sentence)  # 分词
    tagged_sent = pos_tag(tokens)     # 獲取單詞詞性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    sentence_merge =""
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 辭形還原
    for sent in lemmas_sent:
        sentence_merge = sentence_merge+ " "+sent
    return sentence_merge


#============================================================================================================

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = re.sub(r'\d|\n|[^\w]',' ', s)
    tokens = nltk.tokenize.word_tokenize(tokens) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens

#============================================================================================================

def count_score(liquer,tem_score,n): #計算分數
    try:
        total_score = tem_score / n
        return  total_score
    except ZeroDivisionError:
        print("詞句=",liquer)
        a=0
        return a

#============================================================================================================

def get_two_float(f_str, n): #換算小數點幾位
    f_str = str(f_str)      # f_str = '{}'.format(f_str) 也可以轉換為字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 傳入的函數有幾位小數，在字符串後面都添加n為小數0
    return ".".join([a, c])

#============================================================================================================

def load_word2vec_model(words):# 載入模型
    model = word2vec.Word2Vec.load("./word2vec.model")
    # print(model.wv.similar_by_word(words)) #顯示輸入的關鍵字的同義字
    # for co in (model.wv.similar_by_word(words)[:5]): #取出同義字
    x = model.wv.similar_by_word(words)[:3]
    return (x)

#============================================================================================================


def save_word_score(dictdata): # 存單字權重到CSV
    dict_data = {"word": "", "score": ""}
    weight_df = pd.DataFrame(columns=['單字','權重'])
    for k in dictdata:
        dict_data["word"]=k
        dict_data["score"]=dictdata[k]
        weight_list.append(list(dict_data.values()))
    # print(weight_list)
    weight_dff = weight_df.append(pd.DataFrame(weight_list, columns=['單字','權重']))
    weight_dff.to_csv(r'./whisky_weight.csv', index=False, encoding="utf-8-sig")

#============================================================================================================



def execute_multiple_times(): #重複輸入評論
    select = "y"
    while select == "y":
        write_comment()
        select = input("輸入: 繼續 y | 停止 n =")




#============================================================================================================

def old_comment_score(): #現有的評論(1個酒只有一條評論)做分數
    # 存
    df = pd.DataFrame(columns=['酒名', '情緒預測', '總分數'])

    cocktail_list = []
    data = {"name": "", "sentiment": "", "all_score": ""}
    dictdata = {}
    # 開啟 CSV 檔案
    with open('whisky_weight_word2vec_add_again.csv', newline='',encoding='utf8') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            if rows.line_num == 1:
                continue
            else:
                dictdata[row[0]]=row[1]
    # save_word_score(dictdata) #

    s=0
    with open('all_comment.csv', newline='', encoding='utf8') as csvfile:
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            if rows.line_num == 1:
                continue
            else:
                s+=1
                # print(s)
                liquer=row[0]
                data["name"]=liquer

                # print("酒名=",liquer)
                comment=row[1].replace(", ",",")
                # data["all_comment"]=comment
                # print("評論=",comment)
                tokens=my_tokenizer(comment)
                if tokens==[]:
                    data["sentiment"]="Neutral"
                    data["all_score"]=0
                    cocktail_list.append(list(data.values()))

                else:
                    # print(tokens)
                    n=0
                    tem_score = 0
                    for token in tokens : #把單字取出
                        if token in dictdata: #如果單字有在這個字典內就符合的單字值
                            tem_score = tem_score + float(dictdata[token]) #在這行作加減
                            print("比對成功，此單字為:", token, end="")
                            print("，分數:", dictdata[token])
                            n+=1
                        else:  # 如果單字沒有再字典內
                            try:
                                print('此單字[ {} ]不在單字庫，將使用word2vec查詢同義字，代替此單字進行單字庫比對:'.format(token))
                                word2vec_list = load_word2vec_model(token)  # 把單字送進去word2vec 取出前三名同義字
                                synonymous_status = "False"  # 設定同義字檢查狀態為 False
                                sumword = 1
                                for word2 in word2vec_list:  # 從三個同義字一個一個取出
                                    print("Word2vec取出第{}個同義字:{}".format(sumword, word2[0]))
                                    sumword += 1
                                    if synonymous_status == "False":  # 如果同義字檢查狀態為 False 就進入 查詢同義字是否在字典內 ， 如果是 True 就不進去查詢(代表之前已經查詢找到同義字分數)
                                        if word2[0] in dictdata:  # 如果同義字在字典內
                                            tem_score = tem_score + float(dictdata[word2[0]])  # 在這行作加減
                                            print("此同義字[{}]比對單字庫成功，".format(word2[0]), end="")
                                            print("同義字分數:", dictdata[word2[0]])
                                            synonymous_status = "True"
                                            status = word2[0]
                                            n += 1
                                            break
                                        else:
                                            print("此同義字[ {} ]不在單字庫內".format(word2[0]))
                                            if sumword == 4:
                                                print("此單字[ {} ]無同義字比對成功".format(token))
                            except:
                                print("沒有同義字", token)
                                continue

                    total_score = count_score(tokens,tem_score,n)
                    print("總分數=",total_score)
                    all_float_score=get_two_float(total_score, 1)#取小數點 後面數字 是決定幾位數
                    # print("總分數(小數點取3位)=",all_float_score)
                    all_float_score_a=float(all_float_score)
                    data["all_score"]=all_float_score
                    # minmax_score[all_float_score]=liquer #計算min max 分數
                    if all_float_score_a > 0:
                        data["sentiment"]="Positive"
                    elif all_float_score_a == 0:
                        data["sentiment"]="Neutral"
                    else:
                        data["sentiment"]="Negative"
                    # print(data)
                    cocktail_list.append(list(data.values()))

            # print(cocktail_list)
    dff = df.append(pd.DataFrame(cocktail_list, columns=['酒名','情緒預測','總分數']))
    dff.to_csv(r'./sentiment_whisky_final.csv', index=False, encoding="utf-8-sig")

#============================================================================================================




def old_comment_score_one_to_one(): #現有的評論(1個酒多條評論)一條一條做分數
    # 存
    df = pd.DataFrame(columns=['酒名', '情緒預測', '總分數'])

    cocktail_list = []
    data = {"name": "", "sentiment": "", "score": ""}
    dictdata = {}
    # 開啟 CSV 檔案
    with open('whisky_weight_word2vec_add_again.csv', newline='',encoding='utf8') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            if rows.line_num == 1:
                continue
            else:
                dictdata[row[0]]=row[1]
    # save_word_score(dictdata) #

    s=0
    with open('all_user_subnull_trans_emoji.csv', newline='',encoding='utf8') as csvfile:
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            if rows.line_num == 1:
                continue
            else:
                s+=1
                # print(s)
                liquer=row[0]
                data["name"]=liquer

                # print("酒名=",liquer)
                comment=row[2].replace(", ",",")
                # data["all_comment"]=comment
                # print("評論=",comment)
                tokens=my_tokenizer(comment)
                if tokens==[]:
                    data["sentiment"]="Neutral"
                    data["score"]=0
                    cocktail_list.append(list(data.values()))

                else:
                    # print(tokens)
                    n=0
                    tem_score = 0
                    for token in tokens : #把單字取出
                        if token in dictdata: #如果單字有在這個字典內就符合的單字值
                            tem_score = tem_score + float(dictdata[token]) #在這行作加減
                            print("比對成功，此單字為:", token, end="")
                            print("，分數:", dictdata[token])
                            n+=1
                        else:  # 如果單字沒有再字典內
                            try:
                                print('此單字[ {} ]不在單字庫，將使用word2vec查詢同義字，代替此單字進行單字庫比對:'.format(token))
                                word2vec_list = load_word2vec_model(token)  # 把單字送進去word2vec 取出前三名同義字
                                synonymous_status = "False"  # 設定同義字檢查狀態為 False
                                sumword = 1
                                for word2 in word2vec_list:  # 從三個同義字一個一個取出
                                    print("Word2vec取出第{}個同義字:{}".format(sumword, word2[0]))
                                    sumword += 1
                                    if synonymous_status == "False":  # 如果同義字檢查狀態為 False 就進入 查詢同義字是否在字典內 ， 如果是 True 就不進去查詢(代表之前已經查詢找到同義字分數)
                                        if word2[0] in dictdata:  # 如果同義字在字典內
                                            tem_score = tem_score + float(dictdata[word2[0]])  # 在這行作加減
                                            print("此同義字[{}]比對單字庫成功，".format(word2[0]), end="")
                                            print("同義字分數:", dictdata[word2[0]])
                                            synonymous_status = "True"
                                            status = word2[0]
                                            n += 1
                                            break
                                        else:
                                            print("此同義字[ {} ]不在單字庫內".format(word2[0]))
                                            if sumword == 4:
                                                print("此單字[ {} ]無同義字比對成功".format(token))
                            except:
                                print("沒有同義字", token)
                                continue

                    total_score = count_score(tokens,tem_score,n)
                    print("分數=",total_score)
                    all_float_score=get_two_float(total_score, 1)#取小數點 後面數字 是決定幾位數
                    # print("總分數(小數點取3位)=",all_float_score)
                    all_float_score_a=float(all_float_score)
                    data["score"]=all_float_score
                    # minmax_score[all_float_score]=liquer #計算min max 分數
                    if all_float_score_a > 0:
                        data["sentiment"]="Positive"
                    elif all_float_score_a == 0:
                        data["sentiment"]="Neutral"
                    else:
                        data["sentiment"]="Negative"
                    # print(data)
                    cocktail_list.append(list(data.values()))

            # print(cocktail_list)
    dff = df.append(pd.DataFrame(cocktail_list, columns=['酒名','情緒預測','總分數']))
    dff.to_csv(r'./sentiment_whisky_one_to_one_final.csv', index=False, encoding="utf-8-sig")

#============================================================================================================

def write_comment(): #自行輸入評論生成分數
    # 存
    dfdf = pd.DataFrame(columns=['酒名', '情緒預測', '總分數'])

    comment_list = []
    data_comment = {"name": "", "sentiment": "", "all_score": ""}
    dictdata_to_person = {}
    # 開啟 CSV 檔案
    with open('whisky_weight_word2vec_add_again.csv', newline='', encoding='utf8') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            if rows.line_num == 1:
                continue
            else:
                dictdata_to_person[row[0]] = row[1]



    liquer=input("請輸入酒名:")
    data_comment["name"]=liquer
    person_comment = input("請輸入評論:")
    person_comment = part_of_speech_reduction(person_comment)
    tokens = my_tokenizer(person_comment)
    if tokens == []:
        data_comment["sentiment"] = "Neutral"
        data_comment["all_score"] = 0
        comment_list.append(list(data_comment.values()))
    else:
        print(tokens)
        n = 0
        tem_score = 0
        for token in tokens:  # 把單字取出
            if token in dictdata_to_person:  # 如果單字有在這個字典內就符合的單字值
                tem_score = tem_score + float(dictdata_to_person[token])  # 在這行作加減
                print("比對成功，此單字為:",token,end="")
                print("，分數:",dictdata_to_person[token])
                n += 1
            else: #如果單字沒有再字典內
                try:
                    print('此單字[ {} ]不在單字庫，將使用word2vec查詢同義字，代替此單字進行單字庫比對:'.format(token))
                    word2vec_list= load_word2vec_model(token) #把單字送進去word2vec 取出前三名同義字
                    synonymous_status = "False"  # 設定同義字檢查狀態為 False
                    sumword=1
                    for word2 in word2vec_list:  # 從三個同義字一個一個取出
                        print("Word2vec取出第{}個同義字:{}".format(sumword,word2[0]))
                        sumword+=1
                        if synonymous_status == "False":  # 如果同義字檢查狀態為 False 就進入 查詢同義字是否在字典內 ， 如果是 True 就不進去查詢(代表之前已經查詢找到同義字分數)
                            if word2[0] in dictdata_to_person:  # 如果同義字在字典內
                                tem_score = tem_score + float(dictdata_to_person[word2[0]])  # 在這行作加減
                                print("此同義字[{}]比對單字庫成功，".format(word2[0]),end="")
                                print("同義字分數:",dictdata_to_person[word2[0]])
                                synonymous_status = "True"
                                status = word2[0]
                                n += 1
                                break
                            else:
                                print("此同義字[ {} ]不在單字庫內".format(word2[0]))
                                if sumword == 4 :
                                    print("此單字[ {} ]無同義字比對成功".format(token))
                except:
                    print("沒有同義字",token)
                    continue


        print(n)
        total_score = count_score(tokens, tem_score, n)
        # print("總分數=",total_score)
        all_float_score = get_two_float(total_score, 1)  # 取小數點 後面數字 是決定幾位數
        all_float_score_a = float(all_float_score) #分數型態變浮點數
        data_comment["all_score"] = all_float_score

        # minmax_score[all_float_score] = liquer  # 計算min max 分數

        if all_float_score_a > 0:
            data_comment["sentiment"] = "Positive"

        elif all_float_score_a == 0:
            data_comment["sentiment"] = "Neutral"

        else:
            data_comment["sentiment"] = "Negative"

        # print("酒名:", data_comment["name"])
        print("評論:", person_comment)
        print("評論分數:", data_comment["all_score"])
        print("情緒預測:", data_comment["sentiment"])

        comment_list.append(list(data_comment.values()))
        # print(data)
    dfddf = dfdf.append(pd.DataFrame(comment_list, columns=['酒名','情緒預測','總分數']))
    dfddf.to_csv(r'./sentiment_person_score.csv', index=False, encoding="utf-8-sig")

#============================================================================================================

if __name__ == "__main__":
    old_comment_score() #執行把現有評論(酒有多條評論整合成一款酒只有一條)生成分數
    old_comment_score_one_to_one() #執行把現有評論(酒有多條評論)一條一條生成分數
    write_comment() #輸入評論生成分數
    execute_multiple_times() #重複輸入評論生成分數







