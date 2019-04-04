import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# # 有向图
# DG = nx.DiGraph()
# # 添加一个节点
# DG.add_node("A")
# DG.add_node("B")
# #  添加边，有方向，A-->B
# DG.add_edge('A', 'B')
# # 作图，设置节点名显示，节点大小，节点颜色
# nx.draw(DG, with_labels=True, node_size=900, node_color="green")
# plt.show()

# 设置显示中文
# plt.rcParams["font.sans-serif"]=['SimHei']
# plt.rcParams["axes.unicode_minus"]=False
# colors = ["red", "green", 'blue', 'yellow', 'black']
# DG = nx.MultiDiGraph()
# DG.add_node("广西")
# DG.add_node("南宁")
# DG.add_node("北海")
# DG.add_node("桂林")
# DG.add_node("合浦")
# DG.add_edge("广西", "南宁")
# DG.add_edge("广西", "北海")
# DG.add_edge("广西", "桂林")
# DG.add_edge("北海", "合浦")
# nx.draw(DG, with_labels=True, node_size=200, node_color=colors)
# plt.show()

columns = ['std_id','class','name','classroom','label_1','label_2','label_3','label_4','time','label_5']
# print(columns)
df = pd.read_csv("E:\machineLearning\data_visu\data\\nd_course_schedule_info.csv", sep='\t', names=columns)
# print(df)
classes= df['class'].values.tolist()
classrooms=df['classroom'].values.tolist()
nodes = list(set(classes + classrooms))
weights = [(df.loc[index,'class'],df.loc[index,'classroom'])for index in df.index]
weights =  list(set(weights))
# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False
colors = ['red', 'green', 'blue', 'yellow', "black"]
#有向图
DG = nx.DiGraph()
#一次性添加多节点，输入的格式为列表
DG.add_nodes_from(nodes)

DG.add_edges_from(weights)
#作图，设置节点名显示,节点大小，节点颜色
nx.draw(DG,with_labels=True, node_size=1000, node_color=colors)
plt.show()