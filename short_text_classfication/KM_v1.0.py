import jieba
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# 加载停用词
stopwords = pd.read_csv("data/stopwords.txt", quoting=3, encoding='utf8', names=['stopword'], sep='\t', index_col=False)
stopwords = stopwords["stopword"].values

# 加载语料
laogong_df = pd.read_csv("data/beilaogongda.csv", encoding="utf8", sep=',')
laopo_df = pd.read_csv("data/beilaopoda.csv", encoding='utf8', sep=',')
erzi_df = pd.read_csv("data/beierzida.csv", encoding='utf8', sep=',')
nver_df = pd.read_csv("data/beinverda.csv", encoding='utf8', sep=",")

# 数据清洗
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)

laogong = laogong_df["segment"].values.tolist()
laopo = laopo_df["segment"].values.tolist()
erzi = erzi_df["segment"].values.tolist()
nver = nver_df["segment"].values.tolist()
# print(nver)
segments = []
def process_data(data):
    for text in data:
        segs = jieba.lcut(text)
        # 去数字
        segs = [x for x in segs if not str(x).isdigit()]
        # 去左右空格
        segs = [x.strip() for x in segs]
        # 去长度为1的词
        segs = list(filter(lambda x:len(x)>1, segs))
        # 去停用词
        segs = list(filter(lambda x:x not in stopwords, segs))
        segments.append(" ".join(segs))

process_data(laogong)
process_data(laopo)
process_data(erzi)
process_data(nver)

# print(segments)

random.shuffle(segments)

# for x in segments[:10]:
#     print(x)
# 将原始文档集合转换为tf-idf特性矩阵。
# 学习词汇和IDF，返回术语文档矩阵。
# 将计数矩阵转换为标准化的tf或tf idf表示
# 与使用CountVectorizer模块相比，TfidfVectorizer的效果更好。
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
v_f = vectorizer.fit_transform(segments)
# vec = CountVectorizer(
#     analyzer="word",
#     max_features=4000,
# )
# v_f = vec.fit_transform(segments)
print(v_f)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(v_f)
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print(weight)
print("Features length: " + str(len(word)))

numClass=4 #聚类分几簇
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  #这里也可以选择随机初始化init="random"
# 利用数据的奇异值分解进行线性维数约简，将其投影到低维空间
pca = PCA(n_components=10)  # 降维
TnewData = pca.fit_transform(weight)  # 载入N维
s = clf.fit(TnewData)

def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^', 'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for indl in newData[Lab[i]]:
            try:
                y1.append(indl[1])
                x1.append(indl[0])
            except:
                pass
        plt.plot(x1, y1, color[i])
    # 绘制初始中心点
    x1 = []
    y1 = []
    for indl in clf.cluster_centers_:
        try:
            y1.append(indl[1])
            x1.append(indl[0])
        except:
            pass
    plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()

pca = PCA(n_components=2)  # 输出两维
newData = pca.fit_transform(weight)
result = list(clf.predict(TnewData))
plot_cluster(result, newData, numClass)












