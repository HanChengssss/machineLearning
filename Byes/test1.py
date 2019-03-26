datasets = {'banala': {'long': 400, 'not_long': 100, 'sweet': 350, 'not_sweet': 150, 'yellow': 450, 'not_yellow': 50},
            'orange': {'long': 0, 'not_long': 300, 'sweet': 150, 'not_sweet': 150, 'yellow': 300, 'not_yellow': 0},
            'other_fruit': {'long': 100, 'not_long': 100, 'sweet': 150, 'not_sweet': 50, 'yellow': 50,
                            'not_yellow': 150}
            }


def fruit_count(data):
    '''
    计算水果总数
    :param data: 
    :return: 
    '''
    count = 0
    for i in data:
        count += data[i]["long"]
        count += data[i]["not_long"]
    return count

def fruits_count(data):
    '''
    计算各种水果数量
    :param data: 
    :return: 
    '''
    fruits_dic = {}
    for i in data:
        count = data[i]["long"] + data[i]["not_long"]
        fruits_dic[i] = count
    return fruits_dic

def xy_fruit(data):
    '''
    计算水果的先验概率
    :param data: 
    :return: 
    '''
    xy_dic = {}
    count = fruit_count(data)
    f = fruits_count(data)
    for i in data:
        xy_dic[i] = f[i]/count
    return xy_dic

def hy_feature(data):
    hy_fs = {}
    fruits_c = fruits_count(data)
    for fruit in datasets:
        hy_f = {}
        for f in datasets[fruit]:
            hy_f[f] = datasets[fruit][f]/fruits_c[fruit]
        hy_fs[fruit] = hy_f
    return  hy_fs

def xy_feature(data):
    count = fruit_count(data)
    f_dic = {}
    for f in datasets.get("banala"):
        for i in datasets:
            if f not in f_dic:
                f_dic[f] = datasets[i][f]/count
            else:
                f_dic[f] += datasets[i][f]/count
    return f_dic

class navie_bayes_classifier():

    def __init__(self, data):
        self.data = data
        self.count = fruits_count(self.data)
        self.f_count = fruits_count(self.data)
        self.xy_fe = xy_feature(self.data)
        self.xy_fr = xy_fruit(self.data)
        self.hy_fe = hy_feature(self.data)

    def get_result(self, length, sweetness, color):
        fe_list = [length, sweetness, color]
        result_dic = {}
        for fr in self.data:
            xy_fr = self.xy_fr[fr]
            for fe in fe_list:
                xy_fr *= self.hy_fe[fr][fe]
            result_dic[fr] = xy_fr
        return result_dic



count = fruit_count(datasets)
f = fruits_count(datasets)
xy = xy_fruit(datasets)
f_dic = xy_feature(datasets)
h = hy_feature(datasets)
n = navie_bayes_classifier(datasets)
result_dic = n.get_result(length='not_long', sweetness="sweet", color="yellow")
print(result_dic)
