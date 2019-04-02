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

# 提取关键字
# doc2vec
# if-idf
# lda
# textrank

#  构建词袋模型
dictionary = corpora.Dictionary(sentences)  # 构建字典 key：单词 value：对应id
'''
>>> dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])
>>> dct.doc2bow(["this", "is", "máma"])
[(2, 1)]
>>> dct.doc2bow(["this", "is", "máma"], return_missing=True) 显示词典中未包含的词
([(2, 1)], {'is': 1, 'this': 1})
>>> dct.doc2bow(["this", "is", "máma"], return_missing=True, allow_update=True) 将词典中未包含的词更新到词典中
([(2, 1), (5, 1), (6, 1)], {'is': 1, 'this': 1})
'''
# corpus = [dictionary.doc2bow(sentence) for sentence in sentences]  # (0, 2), (1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1),自建词典中该词id和出现的次数
corpus = [dictionary.doc2idx(sentence) for sentence in sentences]  #  [21, 23, 19, 25, 18, 11, 7, 23, 2, 22, 0,] 输出的是自建字典的词的对应id，没有的词id为-1
print(corpus[0])

# lda模型，num_topics是主题个数
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
for topic in lda.print_topics(num_topics=10, num_words=8):
     print(topic[1])

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

num_show_term = 8
num_topics = 10

for i, k in enumerate(range(num_topics)):
    ax = plt.subplot(2, 5, i+1)  # 设置画布
    item_dis_all = lda.get_topic_terms(topicid=k)  # 取出主题
    item_dis = np.array(item_dis_all[: num_show_term])  #取出前8个主题词id和概率 item_dis = array([[11.        ,  0.05263364],[ 6.        ,  0.05263316],[ 0.        ,  0.05263249],[15.        ,  0.0526323 ]])
    #  折线图 x, y, 标记：蓝色 *
    ax.plot(range(num_show_term), item_dis[:, 1], "b*")  # item_dis[:, 1] = array([0.05263364, 0.05263316, 0.05263249, 0.0526323 ])
    item_word_id = item_dis[:, 0].astype(np.int)  # 取出主题词id array([11,  6,  0, 15]) word id
    word = [dictionary.id2token[i] for i in item_word_id]  # word id对应词汇列表
    ax.set_ylabel("概率")
    # 设置点标签名字 ax.text(x,y,名字,风格)
    for j in range(num_show_term):
        ax.text(j, item_dis[j, 1], word[j], bbox=dict(facecolor='green', alpha=0.1))
plt.suptitle("xxxxx")
plt.show()







