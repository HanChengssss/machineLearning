import jieba
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import re
import csv
# 加载停用词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, names=["stopwords"], sep='\t', encoding='utf8')
stopwords = stopwords["stopwords"].values
print(stopwords)

content_lines = [x for x in open("data/zjly_test.csv", 'r', encoding='utf8').readlines()]
# 加载语料
def remove(text):
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    ret = rule.sub("", text)
    return ret

def preprocess_text(content_lines, sentences):
    for content in content_lines:
        con = remove(content)
        segs = jieba.lcut(con)
        segs = list(filter(lambda x:x not in stopwords, segs))
        sentences.append((" ".join(segs)))

sentences = []
preprocess_text(content_lines, sentences)
print(sentences)

vec = joblib.load('vec.pkl')
classifier = joblib.load('classifier.pkl')

with open('data\zjly_result.csv', 'w', encoding='utf8', newline="") as f:
    fieldnames = ["text", "ret"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for sen in sentences:
        s = []
        s.append(sen)
        ret = classifier.predict(vec.transform(s))
        print(sen, " ", ret[0])
        writer.writerow({"text": sen, "ret": ret[0]})

