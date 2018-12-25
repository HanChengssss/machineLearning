#读取文件
import sys
import os
import jieba

def readFile(path):
    fp = open(path, "r", encoding='utf8')
    content = fp.read()
    fp.close()
    return content
#在写一个保存文件的函数，方便我们调用，参数字母也代表是什么意思了,
def readfile(path):
    fp = open(path, "r", encoding='utf8')
    content = fp.read()
    fp.close()
    return content

# 设置输出和输入目录
corpus_path = r"E:\machineLearning\cluster\AMT_SRC/"  # 输入目录
seg_path = r"E:\machineLearning\cluster\cut_word_text/"  # 输出目录
#  获取在未分词语料下的所有子目录
catelist = os.listdir(corpus_path)
print(catelist)


#  结果上看没毛病，接下来开始写我们最重要的部分
#  开始我们的迭代分词
for mydir in catelist:
    class_path = corpus_path + mydir + '/'  #构建出分词文本的目录
    print("class_path==========",class_path)
    seg_dir = seg_path + mydir + '/'  # 构建出输出分词的目录
    if not os.path.exists(seg_dir):  # 是否存在目录，不存在则创建一个
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)  # 获取类别目录下的所有文件
    for file_path in file_list:
        full_name = class_path + file_path#构建出文本的目录作为参数传入我们调用的函数中
        print(full_name) #打印一下分词的本文路径
        content = readfile(full_name).strip()#文本删除前面的空白符
        content = content.replace("'\r\n'", '').replace(""'（）'"",'').strip()  # 删除掉换行和多于的空格
        content_seg=jieba.cut(content)#对文本进行分词
        fp = open(seg_dir+file_path, "w", encoding='utf8')#将文本写入文件中
        for word in content_seg:
            word=' '.join(word)  # 分词后的词语连接空格保存
            fp.write(word)#设置一下我们的编码格式
        fp.close()
    print ("分词结束")