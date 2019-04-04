from gensim.corpora import Dictionary
import gensim
import matplotlib.pyplot as plt
import jieba.analyse as analyse
import jieba
import numpy as np
# 构建词典
sentences = [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"], ["k", "l", "m", "n"], ["o", "p", "q", "r", "s"]]

dics = Dictionary(sentences)

corpus = [dics.doc2bow(sentence) for sentence in sentences]

# print(corpus)

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dics, num_topics=10)
for topic in lda.print_topics(num_topics=5, num_words=4):
    # print(topic[1])  # 分别输出每个主题词下前五个词出现的频率
    pass
top = lda.print_topics()
# print(top)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

num_show_term = 4
num_topics = 5

# 设置画布
# 取出主题
# 取出当前主题的前4个词的id和对应频率
# 转换成arry格式
# 设置折线图
# 取出主题词id
# 取出对应词
# 设置点对应的词
# 设置y轴标题
# 设置标题
# 展示
for i, j in enumerate(range(num_topics)):
    ax = plt.subplot(2, 5, i+1)
    topics = lda.get_topic_terms(j)
    topic_arry = np.array(topics[: num_show_term])
    print(topic_arry)
    ax.plot(range(num_show_term), topic_arry[:, 1], "b*")
    term_ids = topic_arry[:, 0].astype(np.int)
    words = [lda.id2word[i] for i in term_ids]
    for k in range(num_show_term):
        ax.text(k, topic_arry[k, 1], words[k], bbox=dict(facecolor="green", alpha=0.1))
plt.suptitle("xxxxxxxx")
plt.show()








