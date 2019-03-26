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
# stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep='\t', names=["stopwords"], encoding='utf8')
# stopwords = stopwords["stopwords"].values
# print(stopwords)
# 加载语料
# caizheng_df = pd.read_csv("data/caizheng.csv", sep=',', encoding='utf8')
# zichou_df = pd.read_csv("data/zichou.csv", sep=',', encoding='utf8')
# hezi_df = pd.read_csv("data/hezi.csv", sep=',', encoding='utf8')

# 删除语料的nan行
# caizheng_df.dropna(inplace=True)
# zichou_df.dropna(inplace=True)
# hezi_df.dropna(inplace=True)

# 转化为list
# caizheng = caizheng_df.segment.values.tolist()
# zichou = zichou_df.segment.values.tolist()
# hezi  = hezi_df.segment.values.tolist()

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

# sentences = []
# preprocess_text(caizheng, sentences, 0)
# preprocess_text(zichou, sentences, 1)
# preprocess_text(hezi, sentences, 2)
# print(sentences)

# 把数据打散
# random.shuffle(sentences)

# 抽取特征
# vec = CountVectorizer(
#     analyzer="word",
#     max_features=4000,
# )

# 切分语料
# x, y = zip(*sentences)
# print(x)
# print(y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)
# print(x_test)
# 把训练数据转换为词袋模型
# vec.fit(x_train)
# joblib.dump(vec, 'vec.pkl')

# print(vec)
# 算法建模和模型训练
# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB()
# classifier.fit(vec.transform(x_train), y_train)
# # 测试评分
# score = classifier.score(vec.transform(x_test), y_test)
# print(score)
# joblib.dump(classifier, 'classifier.pkl')
vec = joblib.load("vec.pkl")
classifier = joblib.load("classifier.pkl")
x_t = ['部省 补助 投资 和 地方 自筹']
pre = classifier.predict(vec.transform(x_t))

print(pre)

'''
<html>
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
<style type="text/css">
  table tr td { border: 1px solid blue }
  table { border: 1px solid blue }
  span.note { font-size: 9px; color: red }
  div.margin_txt span { display:block;}
</style>
<div id="margin_0" class="margin_txt">
<span id="line_0">现金流量表</span>
<span id="line_1"></span>
<span id="line_2">--</span>
<span id="line_3">会03表</span>
<span id="line_4">编制单位:福州中物建设发展有限公司2017年度货币单位:人民币元</span>
</div>
<table id="table_0">
<tr><td colspan="1" rowspan="1">项目 </td><td colspan="1" rowspan="1">行次</td><td colspan="1" rowspan="1">金额 </td><td colspan="1" rowspan="1">项目 </td><td colspan="1" rowspan="1">行次</td><td colspan="1" rowspan="1">金额 </td></tr>
<tr><td colspan="1" rowspan="1">一、经营活动产生的现金流量:</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">-</td><td colspan="1" rowspan="1">补充资料   </td><td colspan="1" rowspan="1">35</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">销售商品、提供劳务收到的现金</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">50,205,378.44</td><td colspan="1" rowspan="1">1,将净利润调节为经营活动的现金流量:</td><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">收到的税费返还</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">106,088.68</td><td colspan="1" rowspan="1">净利润</td><td colspan="1" rowspan="1">37</td><td colspan="1" rowspan="1">684,996.22</td></tr>
<tr><td colspan="1" rowspan="1">收到的其他与经营活动有关的现金</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">加:计提的资产损失准备</td><td colspan="1" rowspan="1">38</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">现金流入小计</td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">50,311,467.12</td><td colspan="1" rowspan="1">固定资产折旧</td><td colspan="1" rowspan="1">39</td><td colspan="1" rowspan="1">643,641.10</td></tr>
<tr><td colspan="1" rowspan="1">购买商品、接受劳务支付的现金</td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">40,673,714.91</td><td colspan="1" rowspan="1">无形资产摊销</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">-</td></tr>
<tr><td colspan="1" rowspan="1">支付给职工以及为职工支付的现金</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">242,839.91</td><td colspan="1" rowspan="1">长期待摊费用摊销</td><td colspan="1" rowspan="1">41</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">支付的各项税费</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">809,789.37</td><td colspan="1" rowspan="1">待摊费用减少(减:增加)</td><td colspan="1" rowspan="1">42</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">支付的其他与经营活动有关的现金</td><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">13,228,155.90</td><td colspan="1" rowspan="1">预提费用增加(减:减少)</td><td colspan="1" rowspan="1">43</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">现金流出小计</td><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">54,954,500.09</td><td colspan="1" rowspan="1">处置固定资产、无形资产和其他长期资产的损失(减:收益)</td><td colspan="1" rowspan="1">44</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">经营活动产生的现金流量净额</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">-4,643,032.97</td><td colspan="1" rowspan="1">固定资产报废损失</td><td colspan="1" rowspan="1">45</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">二、投资活动产生的现金流量</td><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">财务费用</td><td colspan="1" rowspan="1">46</td><td colspan="1" rowspan="1">4,284.78</td></tr>
<tr><td colspan="1" rowspan="1">收回投资所收到的现金</td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">投资损失(减:收益)</td><td colspan="1" rowspan="1">47</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">取得投资收益所收到的现金</td><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">递延税款贷项(减:借项)</td><td colspan="1" rowspan="1">48</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">处置固定资产,无形资产和其他长期资产而收到的现金净额</td><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">存货的减少(减:增加)</td><td colspan="1" rowspan="1">49</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">收到的其他与投资活动有关的现金</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">经营性应收项目的减少(减:增加)</td><td colspan="1" rowspan="1">50</td><td colspan="1" rowspan="1">-6,452,639.83</td></tr>
<tr><td colspan="1" rowspan="1">现金流入小计</td><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">经营性应付项目的增加(减:减少)</td><td colspan="1" rowspan="1">51</td><td colspan="1" rowspan="1">476,684.76</td></tr>
<tr><td colspan="1" rowspan="1">购建固定资产、无形资产和其他长期资产所支付的现金</td><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">83,200.00</td><td colspan="1" rowspan="1">其他</td><td colspan="1" rowspan="1">52</td><td colspan="1" rowspan="1">0.00</td></tr>
<tr><td colspan="1" rowspan="1">投资所支付的现金</td><td colspan="1" rowspan="1">19</td><td colspan="1" rowspan="1">-</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">53</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">支付的其他与投资活动有关的现金</td><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">-</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">54</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">现金流出小计</td><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">83,200.00</td><td colspan="1" rowspan="1">经营活动产生的现金流量净额</td><td colspan="1" rowspan="1">55</td><td colspan="1" rowspan="1">-4,643,032.97</td></tr>
<tr><td colspan="1" rowspan="1">投资活动产生的现金流量净额</td><td colspan="1" rowspan="1">22</td><td colspan="1" rowspan="1">-83,200.00</td><td colspan="1" rowspan="1">2、不涉及现金收支的投资和筹资活动:</td><td colspan="1" rowspan="1">56</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">二、筹资活动产生的现金流量:</td><td colspan="1" rowspan="1">23</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">债务转为资本</td><td colspan="1" rowspan="1">57</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">吸收投资所受到的现金</td><td colspan="1" rowspan="1">24</td><td colspan="1" rowspan="1">5,000,000.00</td><td colspan="1" rowspan="1">一年内到期的可转换公司债券</td><td colspan="1" rowspan="1">58</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">借款所收到的现金</td><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">负资租入固定资产</td><td colspan="1" rowspan="1">59</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">收到的其他与筹资活动有关的现金</td><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">-</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">60</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">现金流入小计</td><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">5,000,000.00</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">61</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">偿还债务所支付的现金</td><td colspan="1" rowspan="1">28</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">62</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">分配股利、利润和偿付利息所支付的现金</td><td colspan="1" rowspan="1">29</td><td colspan="1" rowspan="1">4,284.78</td><td colspan="1" rowspan="1">3,现金及现金等价物净增加情况:</td><td colspan="1" rowspan="1">63</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">支付的其他与筹资活动有关的现金</td><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">现金的期末余额</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">7,356,964.31</td></tr>
<tr><td colspan="1" rowspan="1">现金流出小计</td><td colspan="1" rowspan="1">31</td><td colspan="1" rowspan="1">4,284.78</td><td colspan="1" rowspan="1">减:现金的期初余额</td><td colspan="1" rowspan="1">65</td><td colspan="1" rowspan="1">7,087,482.06</td></tr>
<tr><td colspan="1" rowspan="1">筹资活动产生的现金流量净额</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1">4,995,715.22</td><td colspan="1" rowspan="1">加:现金等价物的期末余额 </td><td colspan="1" rowspan="1">66</td><td colspan="1" rowspan="1">-</td></tr>
<tr><td colspan="1" rowspan="1">四、汇率变动对现金的影响额</td><td colspan="1" rowspan="1">33</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">减:现金净价物的期初余额</td><td colspan="1" rowspan="1">67</td><td colspan="1" rowspan="1"></td></tr>
<tr><td colspan="1" rowspan="1">五、现金及现金等价物净增加额</td><td colspan="1" rowspan="1">34</td><td colspan="1" rowspan="1">269,482.25</td><td colspan="1" rowspan="1">现金及现金等价物净增加额</td><td colspan="1" rowspan="1">68</td><td colspan="1" rowspan="1">269,482.25</td></tr>
</table>
<div id="margin_1" class="margin_txt">
<span id="line_0"></span>
</div>
</html>

'''