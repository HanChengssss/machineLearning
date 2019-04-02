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
    topic_item = lda.get_topic_terms(topicid=j)
    arry_topic = np.array(topic_item[: 4])
    print(arry_topic)
    ax.plot(range(num_show_term), arry_topic[:, 1], "b*")
    all_term_id = arry_topic[:, 0].astype(np.int)
    word_list = [dics.id2token[i] for i in all_term_id]
    ax.set_ylabel("概率")
    for k in range(num_show_term):
        ax.text(k, arry_topic[k,1], word_list[k], bbox=dict(facecolor="green", alpha=0.1))
plt.suptitle("xxxxxxx")
plt.show()






