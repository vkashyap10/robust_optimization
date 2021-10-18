import numpy as np
import pandas as pd

data = pd.read_csv("all_stocks_5yr.csv")
data['date'] = data['date'].str.replace("-","")
data['date'] = data['date'].astype(int)
data = np.array(data)
valid = np.logical_and(data[:,0] >= 20150000,data[:,0] <= 20180000)
data = data[valid]
data = data[:75500,1]
print(data[754])
data = data.reshape(100,755)
print(data)
print(data[46,:])
print(data[:,0])

np.save("first100.npy",data)