import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("./data/youbike_data5.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.duplicated().sum())

data["available_rent_bikes"] = data["available_rent_bikes"].fillna(data["available_rent_bikes"].mean())
print(data.isnull().sum())

print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print(data.duplicated().sum())

sns.boxenplot(data['available_rent_bikes'])
plt.show()