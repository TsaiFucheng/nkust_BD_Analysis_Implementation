# =============================================================================
# 載入所需的 Python 模組
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')  # 忽略警告訊息

# =============================================================================
# 環境設定
# =============================================================================
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")  # 設定 seaborn 樣式

# =============================================================================
# 載入並準備資料
# =============================================================================
df_titanic = sns.load_dataset("titanic")

columns_t = ['生還', '艙等', '性別', '年齡', '手足與配偶數',
             '父母與子女數', '票價', '乘船港代碼', '艙等名', '男女兒童',
             '成人男子', '甲板', '乘船港', '生還與否', '單身']

df_titanic.columns = columns_t

print("=" * 80)
print("Titanic 資料集進階分析")
print("=" * 80)
print(f"資料筆數: {len(df_titanic)}")
print(f"欄位數量: {len(df_titanic.columns)}")
print("=" * 80)

# =============================================================================
# 1. 散佈圖分析 (Scatter Plot Analysis)
# =============================================================================
print("\n【練習 1】散佈圖分析")
print("-" * 80)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.scatter(df_titanic['年齡'], df_titanic['票價'], alpha=0.5)
plt.xlabel('年齡')
plt.ylabel('票價')
plt.title('年齡 vs 票價')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
colors = df_titanic['生還'].map({0: 'red', 1: 'green'})
plt.scatter(df_titanic['年齡'], df_titanic['票價'], c=colors, alpha=0.5)
plt.xlabel('年齡')
plt.ylabel('票價')
plt.title('年齡 vs 票價（依生還狀況著色）')
plt.legend(['死亡 (紅)', '生還 (綠)'], loc='upper right')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
for pclass in [1, 2, 3]:
    mask = df_titanic['艙等'] == pclass
    plt.scatter(df_titanic[mask]['年齡'], df_titanic[mask]['票價'],
                alpha=0.5, label=f'{pclass}等艙')
plt.xlabel('年齡')
plt.ylabel('票價')
plt.title('年齡 vs 票價（依艙等分組）')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
for sex in df_titanic['性別'].unique():
    mask = df_titanic['性別'] == sex
    plt.scatter(df_titanic[mask]['年齡'], df_titanic[mask]['票價'],
                alpha=0.5, label=sex)
plt.xlabel('年齡')
plt.ylabel('票價')
plt.title('年齡 vs 票價（依性別分組）')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ 散佈圖已繪製完成")

# =============================================================================
# 2. 進階統計分析 (Advanced Statistics)
# =============================================================================
print("\n【練習 2】進階統計分析")
print("-" * 80)

print("\n2.1 數值欄位的相關係數矩陣")
numeric_cols = ['生還', '艙等', '年齡', '手足與配偶數', '父母與子女數', '票價']
correlation_matrix = df_titanic[numeric_cols].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', square=True, linewidths=1)
plt.title('數值欄位相關係數熱力圖')
plt.tight_layout()
plt.show()

print("\n2.2 性別 vs 生還的交叉表")
crosstab_sex_survived = pd.crosstab(df_titanic['性別'], df_titanic['生還與否'],
                                     margins=True)
print(crosstab_sex_survived)

crosstab_sex_survived_pct = pd.crosstab(df_titanic['性別'], df_titanic['生還與否'],
                                         normalize='index') * 100
crosstab_sex_survived_pct.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('各性別的生還率 (%)')
plt.xlabel('性別')
plt.ylabel('百分比 (%)')
plt.xticks(rotation=0)
plt.legend(title='生還與否')
plt.tight_layout()
plt.show()

print("\n2.3 艙等 vs 生還的交叉表")
crosstab_class_survived = pd.crosstab(df_titanic['艙等名'], df_titanic['生還與否'],
                                       margins=True)
print(crosstab_class_survived)

print("\n2.4 各組別的生還率")
print("\n依性別分組:")
survival_by_sex = df_titanic.groupby('性別')['生還'].mean() * 100
print(survival_by_sex.to_frame('生還率 (%)'))

print("\n依艙等分組:")
survival_by_class = df_titanic.groupby('艙等名')['生還'].mean() * 100
print(survival_by_class.to_frame('生還率 (%)'))

print("\n依乘船港分組:")
survival_by_embark = df_titanic.groupby('乘船港')['生還'].mean() * 100
print(survival_by_embark.to_frame('生還率 (%)'))

# =============================================================================
# 3. 資料清理實作 (Data Cleaning)
# =============================================================================
print("\n【練習 3】資料清理實作")
print("-" * 80)

df_cleaned = df_titanic.copy()

print("\n3.1 缺失值統計")
missing_values = df_cleaned.isnull().sum()
missing_percentage = (missing_values / len(df_cleaned)) * 100
missing_df = pd.DataFrame({
    '缺失數量': missing_values,
    '缺失比例 (%)': missing_percentage
})
print(missing_df[missing_df['缺失數量'] > 0].sort_values('缺失數量', ascending=False))

print("\n3.2 處理缺失值")

age_median = df_cleaned['年齡'].median()
df_cleaned['年齡'].fillna(age_median, inplace=True)
print(f"✓ 年齡缺失值已用中位數 ({age_median:.2f}) 填補")

embark_mode = df_cleaned['乘船港'].mode()[0]
df_cleaned['乘船港'].fillna(embark_mode, inplace=True)
print(f"✓ 乘船港缺失值已用眾數 ({embark_mode}) 填補")

df_cleaned['甲板'].fillna('Unknown', inplace=True)
print("✓ 甲板缺失值已標記為 'Unknown'")

if df_cleaned['票價'].isnull().any():
    fare_mean = df_cleaned['票價'].mean()
    df_cleaned['票價'].fillna(fare_mean, inplace=True)
    print(f"✓ 票價缺失值已用平均值 ({fare_mean:.2f}) 填補")

print(f"\n清理後剩餘缺失值: {df_cleaned.isnull().sum().sum()}")

print("\n3.3 異常值檢測（票價）")

Q1 = df_cleaned['票價'].quantile(0.25)
Q3 = df_cleaned['票價'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_cleaned[(df_cleaned['票價'] < lower_bound) |
                       (df_cleaned['票價'] > upper_bound)]
print(f"票價異常值數量: {len(outliers)}")
print(f"異常值範圍: < {lower_bound:.2f} 或 > {upper_bound:.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df_cleaned['票價'].dropna())
plt.title('票價箱型圖')
plt.ylabel('票價')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
df_cleaned.boxplot(column='票價', by='艙等名', figsize=(12, 5))
plt.title('各艙等的票價分布')
plt.suptitle('')
plt.xlabel('艙等')
plt.ylabel('票價')
plt.tight_layout()
plt.show()

# =============================================================================
# 4. 複合圖表 (Advanced Visualizations)
# =============================================================================
print("\n【練習 4】複合圖表")
print("-" * 80)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
df_cleaned.boxplot(column='年齡', by='艙等名', ax=plt.gca())
plt.title('各艙等的年齡分布（箱型圖）')
plt.suptitle('')
plt.xlabel('艙等')
plt.ylabel('年齡')

plt.subplot(2, 2, 2)
sns.violinplot(data=df_cleaned, x='生還與否', y='年齡', ax=plt.gca())
plt.title('年齡分布（按生還狀況）')
plt.xlabel('生還與否')
plt.ylabel('年齡')

plt.subplot(2, 2, 3)
sns.violinplot(data=df_cleaned, x='艙等名', y='票價', ax=plt.gca())
plt.title('票價分布（按艙等）')
plt.xlabel('艙等')
plt.ylabel('票價')
plt.xticks(rotation=15)

plt.subplot(2, 2, 4)
crosstab_class = pd.crosstab(df_cleaned['艙等名'], df_cleaned['生還與否'])
crosstab_class.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('各艙等的生還/死亡人數')
plt.xlabel('艙等')
plt.ylabel('人數')
plt.xticks(rotation=0)
plt.legend(title='生還與否')

plt.tight_layout()
plt.show()

print("✓ 複合圖表已繪製完成")

print("\n繪製配對圖（Pair Plot）...")
numeric_subset = ['生還', '艙等', '年齡', '票價']
pairplot = sns.pairplot(df_cleaned[numeric_subset + ['性別']],
                         hue='性別', diag_kind='kde',
                         plot_kws={'alpha': 0.6}, height=2.5)
pairplot.fig.suptitle('數值欄位配對圖（依性別著色）', y=1.02)
plt.show()

print("✓ 配對圖已繪製完成")

# =============================================================================
# 總結統計
# =============================================================================
print("\n" + "=" * 80)
print("進階分析總結")
print("=" * 80)
print(f"原始資料筆數: {len(df_titanic)}")
print(f"清理後資料筆數: {len(df_cleaned)}")
print(f"整體生還率: {df_cleaned['生還'].mean() * 100:.2f}%")
print(f"男性生還率: {df_cleaned[df_cleaned['性別'] == 'male']['生還'].mean() * 100:.2f}%")
print(f"女性生還率: {df_cleaned[df_cleaned['性別'] == 'female']['生還'].mean() * 100:.2f}%")
print(f"一等艙生還率: {df_cleaned[df_cleaned['艙等'] == 1]['生還'].mean() * 100:.2f}%")
print(f"二等艙生還率: {df_cleaned[df_cleaned['艙等'] == 2]['生還'].mean() * 100:.2f}%")
print(f"三等艙生還率: {df_cleaned[df_cleaned['艙等'] == 3]['生還'].mean() * 100:.2f}%")
print("=" * 80)
print("所有進階練習已完成！")
print("=" * 80)
