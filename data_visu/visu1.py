import jieba
import pandas as pd
import numpy as np
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

dir = "data//"
# 定义语料文件路径
file = "".join([dir, "beilaogongda.csv"])
# 定义停用词文件路径
stop_word = "".join([dir, "stopwords.txt"])
# 加载停用词
stopwords = pd.read_csv(stop_word, encoding='utf8', quoting=3, sep='\t', index_col=False, names=["stopword"])
stopwords = stopwords["stopword"].values
# 定义wordcloud中字体文件的路径
simhei = "".join([dir, "simhei.ttf"])
# 读取语料
df = pd.read_csv(file, encoding='utf-8')
# print(df)
df.head()
# 如果存在nan，删除
df.dropna(inplace=True)
# 将content转化为list
content = df.segment.values.tolist()
# print(content)

segment = []
# 数据清洗
for line in content:
    try:
        # 分词
        segs = jieba.lcut(line)
        # 去数字
        segs = [x for x in segs if not str(x).isdigit()]
        # 去左右空格
        segs = list(filter(lambda x: x.strip(), segs))
        # 去停用词
        segs = list(filter(lambda x: x not in stopwords, segs))
        segment.append(segs)
    except:
        print(line)
        continue
# 分词后加入一个新的dataframe
# words_df = pd.DataFrame({"segment": segment})
# print(words_df)
words_df = pd.DataFrame({'Animal' : ['Falcon', 'Falcon','Parrot', 'Parrot'], 'Max Speed' : [380., 370., 24., 26.]})
# print(words_df)
# 按照关键字groupby分组统计词频，并按照计数降序排序
# words_stat = words_df.groupby(by=["segment"])["segment"].agg({"计数": np.size})
words_stat = words_df.groupby(by=["Animal"])
print(words_stat)
# words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)
#
# # 第一种方式
# wordcloud = WordCloud(font_path=simhei, background_color="write", max_font_size=80)
# word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
# wordcloud = wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)

# 第二种方式
text = " ".join(words_stat["Animal"].head(100).astype(str))
abel_mask = imread(r"data//china.jpg")
wordcloud2 = WordCloud(background_color="white",
                       mask=abel_mask,
                       max_words=3000,
                       font_path=simhei,
                       width=4.0,
                       max_font_size=300,
                       random_state=42).generate(text)

# 根据图片生成词云颜色
image_colors = ImageColorGenerator(abel_mask)
wordcloud2.recolor(color_func=image_colors)
# 显示图片
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
wordcloud2.to_file(r"wordcloud2.jpg")