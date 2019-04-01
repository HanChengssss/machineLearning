import jieba.analyse as analyse
import numpy as np
import pandas as pd
import jieba
from gensim import corpora
import gensim
import matplotlib.pyplot as plt
stopwords = pd.read_csv("data/stopwords.txt", quoting=3, encoding='utf8', sep='\t', index_col=False, names=["stopword"])
stopwords = stopwords["stopword"].values

# 加载语料
df = pd.read_csv("data/car.csv", encoding='utf8')
# print(df)
df.dropna(inplace=True)
lines = df.content.values.tolist()
# print(lines)

sentences = []
for line in lines:
    try:
        segs = jieba.lcut(line)
        segs = [x for x in segs if not str(x).isdigit()]
        segs = list(filter(lambda x:x.strip(), segs))
        segs = list(filter(lambda x:x not in stopwords, segs))
        sentences.append(segs)
    except:
        print(line)
        continue

#  构建词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
# lda模型，num_topics是主题个数
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
for topic in lda.print_topics(num_topics=10, num_words=8):
     print(topic[1])









