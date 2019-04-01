import jieba.analyse as analyse
import jieba
import pandas as pd
from gensim import corpora, models, similarities
import gensim
import numpy as np
import matplotlib.pyplot as plt

stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep='\t', names=["stopword"], encoding='utf8')
stopwords = stopwords["stopword"].values
print(stopwords)




