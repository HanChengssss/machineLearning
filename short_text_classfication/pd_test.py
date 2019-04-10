import pandas as pd
import csv
df = pd.read_csv("E:\machineLearning\short_text_classfication\data\crm_proj.csv", sep='	', encoding='utf8')
proj_names = df["工程名称"].values.tolist()
tender_names = df["受益人"].values.tolist()
print(len(proj_names))
print(len(tender_names))

with open("crm_prj.csv", 'w', encoding='utf8') as f:
    for i in range(len(proj_names)):
        f.write(proj_names[i] + "," + tender_names[i] + "\n")


