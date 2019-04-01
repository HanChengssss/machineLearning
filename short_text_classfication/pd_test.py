import pandas as pd

df = pd.read_csv("data/pd_test.csv", sep=',', quoting=3, encoding='utf8')
# print(df)

df_t = df.segment.values.tolist()
# print(df_t[0])

# print(list(filter(lambda x:x.strip(), df_t)))

f = lambda x:x not in [1,2,3]
# ret = f(4)
r = filter(f, [2, 4, 6])
print()