import pandas as pd
data = pd.read_csv("./data/youbike.csv")
# print(data.head())
# print(data.info())
print(data.isnull().sum())