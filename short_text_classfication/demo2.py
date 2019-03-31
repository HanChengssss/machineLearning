import jieba
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# 处理文本
# with open('data/hezi.csv', 'r', encoding='utf8') as f:
#     lines = f.readlines()
# with open("data/hezi.csv", 'w', encoding='utf8') as f:
#     for line in lines:
#         l = line.strip().split("	")
#         # print(l)
#         s = ",".join(l)
#         f.write(s + '\n')

# 加载停用词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep='\t', names=["stopwords"], encoding='utf8')
stopwords = stopwords["stopwords"].values
print(stopwords)
# 加载语料
caizheng_df = pd.read_csv("data/caizheng.csv", sep=',', encoding='utf8')
zichou_df = pd.read_csv("data/zichou.csv", sep=',', encoding='utf8')
hezi_df = pd.read_csv("data/hezi.csv", sep=',', encoding='utf8')

# 删除语料的nan行
caizheng_df.dropna(inplace=True)
zichou_df.dropna(inplace=True)
hezi_df.dropna(inplace=True)

# 转化为list
caizheng = caizheng_df.segment.values.tolist()
zichou = zichou_df.segment.values.tolist()
hezi  = hezi_df.segment.values.tolist()

# 封装分词去停用词函数
def preprocess_text(content_lines, sentences, categroy):
    '''
    :param content_lines: 需要处理的语料
    :param sentences: 用来存放分好的词
    :param category: 标签类型
    :return: 
    '''
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            # 去数字
            segs = [v for v in segs if not str(v).isdigit()]

            segs = list(filter(lambda x:x.strip(), segs))

            segs = list(filter(lambda x:len(x) > 1, segs))

            segs = list(filter(lambda x:x not in stopwords, segs))

            sentences.append((" ".join(segs), categroy))
        except:
            print(line)
            continue

sentences = []
preprocess_text(caizheng, sentences, 0)
preprocess_text(zichou, sentences, 1)
preprocess_text(hezi, sentences, 2)
# print(sentences)

# 把数据打散
random.shuffle(sentences)

# 抽取特征
vec = CountVectorizer(
    analyzer="word",
    max_features=4000,
)

# 切分语料
x, y = zip(*sentences)
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)
print(x_test)
# 把训练数据转换为词袋模型
vec.fit(x_train)
# joblib.dump(vec, 'vec.pkl')

# print(vec)
# 算法建模和模型训练
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
classifier1 = MultinomialNB()
classifier2 = BernoulliNB()
classifier1.fit(vec.transform(x_train), y_train)
classifier2.fit(vec.transform(x_train), y_train)
# 测试评分
score1 = classifier1.score(vec.transform(x_test), y_test)
score2 = classifier2.score(vec.transform(x_test), y_test)
print(score1)
print(score2)
# 0.75 伯努利
# 0.8611111111111112 多项分布
# joblib.dump(classifier, 'classifier.pkl')
# vec = joblib.load("vec.pkl")
# classifier = joblib.load("classifier.pkl")
# x_t = ['部省 补助 投资 和 地方 自筹']
# pre = classifier.predict(vec.transform(x_t))
#
# print(pre)