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
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

num_show_term = 8
num_topics = 10
a = lda.get_topic_terms(topicid=1)
print(a)

for i, k in enumerate(range(num_topics)):
    ax = plt.subplot(2, 5, i+1)
    item_dis_all = lda.get_topic_terms(topicid=k)
    item_dis = np.array(item_dis_all[: num_show_term])
    ax.plot(range(num_show_term), item_dis[:, 1], "b*")
    item_word_id = item_dis[:, 0].astype(np.int)
    word = [dictionary.id2token[i] for i in item_word_id]
    ax.set_ylabel("概率")
    for j in range(num_show_term):
        ax.text(j, item_dis[j, 1], word[j], bbox=dict(facecolor='green', alpha=0.1))
plt.suptitle("xxxxx")
plt.show()






