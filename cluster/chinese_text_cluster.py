import jieba
import os
import numpy as np
import pymysql
import random
from numpy import *

def read_from_file(file_name):
    '''
    :param file_name: 文件名称
    :return: 读取文件内容
    '''
    with open(file_name, "r", encoding='utf8') as fp:
        words = fp.read()
    return words


def stop_words(stop_word_file):
    '''
    :param stop_word_file: 忽略词文件名
    :return: 忽略词集合
    '''
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    return set(new_words)


def del_stop_words(words, stop_words_set):
    #   words是读取的文本内容
    #   返回的会是去除停用词后的文档
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stop_words_set:
            new_words.append(r)
    return new_words

def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        for r in data:
            if r.strip():
                f.write(r.strip() + "\n")

def get_text_data():
    conn = pymysql.connect(host="127.0.0.1", user="root", password="spider_hc", db="python_db", charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
    try:
        sql = "SELECT AMT_SRC FROM zhaobiao_table"
        with conn.cursor() as cursor:
            cursor.execute(sql)
        result = cursor.fetchall()
        with open("E:\machineLearning\cluster\\text\AMT_SRC.csv", 'a', encoding='utf8') as f:
            for r in result:
                if r.get("AMT_SRC"):
                    f.write(r.get("AMT_SRC").strip() + "\n")
    finally:
        print("mysqlClose")
        conn.close()


def get_all_vector(file_path, stop_words_set):
    # 读取所有的字段列表
    with open(file_path, 'r', encoding='utf8') as f:
        posts = f.readlines()
    docs = []
    word_set = set()
    # 删除所有字段里面的忽略词
    for post in posts:
        # print(post.strip())
        doc = del_stop_words(post.strip(), stop_words_set)
        docs.append(doc)
        word_set |= set(doc)
        #print len(doc),len(word_set)

    word_set = list(word_set)
    docs_vsm = []
    # 统计所有文本中所有词出现的频率
    for doc in docs:
        temp_vector = []
        for word in word_set:
            temp_vector.append(doc.count(word) * 1.0)
        # print(temp_vector)
        docs_vsm.append(temp_vector)

    docs_matrix = np.array(docs_vsm)  # 转化成队列
    # nonzero 返回非零元素的索引
    # column_sum 非零列数
    column_sum = [float(len(np.nonzero(docs_matrix[:, i])[0])) for i in range(docs_matrix.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    # 求每个词的idf值，是一个矩阵
    idf = np.log(column_sum)
    idf = np.diag(idf)
    print(idf)
    # 请仔细想想，根绝IDF的定义，计算词的IDF并不依赖于某个文档，所以我们提前计算好。
    # 注意一下计算都是矩阵运算，不是单个变量的运算。
    for doc_v in docs_matrix:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())
        # dot()返回的是两个数组的点积(dot product)
    tfidf = np.dot(docs_matrix, idf)
    '''
    列是所有文档总共的词的集合。
    每行代表一个文档。
    每行是一个向量，向量的每个值是这个词的权值。
    '''
    return posts, tfidf


def gen_sim(A, B):
    num = float(np.dot(A,B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n))) # create centroid mat
    for j in range(n): # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids


def kMeans(dataSet, k, distMeas=gen_sim, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) # create mat to assign data points
                                      # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    counter = 0
    while counter <= 50:
        counter += 1
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] # get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment


if __name__ == '__main__':
    stop_word_set = stop_words("E:\machineLearning\cluster\\text\stop_words.csv")
    # words = read_from_file("E:\machineLearning\cluster\\text\AMT_SRC.csv")
    # new_words = del_stop_words(words, stop_word_set)
    # write_to_file("E:\machineLearning\cluster\\text\AMT_SRC_cut.csv", new_words)
    posts, tfidf = get_all_vector("E:\machineLearning\cluster\\text\AMT_SRC.csv", stop_word_set)
    # print(tfidf)
    myCentroids, clustAssing = kMeans(tfidf, 3, gen_sim, randCent)
    for label, name in zip(clustAssing[:, 0], posts):
        print(label, name)

