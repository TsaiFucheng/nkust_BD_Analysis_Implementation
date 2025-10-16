import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from IPython.display import display





np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
plt.rcParams["font.size"] = 14



df_titanic = sns.load_dataset("titanic")
# print(df_titanic.head())

columns_t = ['生還', '艙等', '性別', '年齡', '手足與配偶數',
             '父母與子女數', '票價', '乘船港代碼', '艙等名', '男女兒童',
             '成人男子', '甲板', '乘船港', '生還與否', '單身']

df_titanic.columns = columns_t

# print(df_titanic.head())

# print(df_titanic.isnull().sum())

# print(df_titanic['乘船港'].value_counts())
# print(df_titanic['生還與否'].value_counts())

# display(df_titanic.describe())
# display(df_titanic.groupby("性別").mean(numeric_only=True))

# columns_n = ['生還', '艙等', '年齡', '手足與配偶數', '父母與子女數', '票價']
# plt.rcParams['figure.figsize'] = (10,10)
# df_titanic[columns_n].hist()
# plt.show()

columns_c = ["性別","乘船港","艙等名","成人男子"]
plt.rcParams['figure.figsize'] = (8,8)
for i,name in enumerate(columns_c):
    ax = plt.subplot(2,2,i+1)
    df_titanic[name].value_counts().plot(kind='bar',title=name,ax=ax)

plt.tight_layout()
plt.show()