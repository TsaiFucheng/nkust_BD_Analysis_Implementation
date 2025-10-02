import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("./data/youbike.csv")
# print(data.head())
# print(data.info())
# print(data.isnull().sum())
# print(data.duplicated().sum())

sns.boxenplot(data['available_rent_bikes'])
plt.show()