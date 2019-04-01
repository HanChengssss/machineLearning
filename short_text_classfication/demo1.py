import jieba
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# 加载停用词

stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep='\t', names=["stopword"], encoding='utf8')
stopwords = stopwords["stopword"].values
# print(stopwords)

# 加载语料提取分词
laogong_df = pd.read_csv('data/beilaogongda.csv', encoding='utf-8', sep=',')

# print(laogong_df)
laopo_df = pd.read_csv("data/beilaopoda.csv", sep=',', encoding='utf8')
erzi_df = pd.read_csv("data/beierzida.csv", sep=',', encoding='utf8')
nver_df = pd.read_csv("data/beinverda.csv", sep=',', encoding='utf8')

# 删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
print(laogong_df)
# 转换为list
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()
print(laogong)

# 分词和去停用词
def preprocess_text(content_lines, sentences, category):
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
            # 去左右空格
            segs = list(filter(lambda x:x.strip(), segs))
            # 去长度为1的字符
            segs = list(filter(lambda x:len(x) > 1, segs))
            # 去掉停用词
            segs = list(filter(lambda x:x not in stopwords, segs))
            # 打标签
            # print(" ".join(segs), category)
            sentences.append((" ".join(segs), category))
            # print(segs)
            # print(sentences)
        except:
            print(line)
            continue

sentences = []
preprocess_text(laogong, sentences, 0)
preprocess_text(laopo, sentences, 1)
preprocess_text(erzi, sentences, 2)
preprocess_text(nver, sentences, 3)

# print(sentences)

# 把数据打散
random.shuffle(sentences)
# for i in range(10):
#     print(sentences[i])

# 抽取特征

vec = CountVectorizer(
    analyzer="word",
    max_features=4000,
)

# 把语料切分
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)
# print(x_test)
# print(y_test)

# 把训练数据转化为词袋模型
vec.fit(x_train)
feature_names = vec.get_feature_names()
# an = vec.analyzer
# print(feature_names)
x = vec.fit_transform(x_train)
print(x.toarray())

# print(x_train)
# 算法建模和模型训练
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
# print(vec.transform(x_test))  # (0, 62)	1 第0个列表元素，**词典中索引为62的元素**， 词频
# 测试评分
score = classifier.score(vec.transform(x_test), y_test)
print(score)
pre = classifier.predict(vec.transform(x_test))

test_result = []
for i in range(len(x_test)):
    j = x_test[i]
    k = y_test[i]
    l = pre[i]
    test_result.append((j,k,l))
# print(test_result)
# 0.9952267303102625
# 0.9928400954653938

