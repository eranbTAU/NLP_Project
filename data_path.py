import os
import pandas as pd

dir = './data/ireland-news-headlines.csv'

data_csv = []
for r, d, f in os.walk(os.path.join(dir, classes)):


print('finish')

d, c, id = [], [] ,[]
for dat in data_csv:
    c += dat[0]
    d += dat[1]
    id += dat[2]

data_all = pd.DataFrame([id,d,c]).transpose()
save_path = r'/home/eranbamani/Documents/LongeRange_CSV'
name_file = 'data_all.csv'
path = os.path.join(save_path, name_file)
data_all.to_csv(path, sep='\t', index=False)


nRowsRead = 10000
data = pd.read_csv(dir, delimiter=',')
