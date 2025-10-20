# =============================================================================
# 載入所需的 Python 模組
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 環境設定
# =============================================================================
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

# =============================================================================
# 載入並準備資料
# =============================================================================
print("=" * 80)
print("Titanic 資料集進階預處理技術")
print("=" * 80)

df_titanic = sns.load_dataset("titanic")

columns_t = ['生還', '艙等', '性別', '年齡', '手足與配偶數',
             '父母與子女數', '票價', '乘船港代碼', '艙等名', '男女兒童',
             '成人男子', '甲板', '乘船港', '生還與否', '單身']

df_titanic.columns = columns_t

print(f"資料筆數: {len(df_titanic)}")
print(f"欄位數量: {len(df_titanic.columns)}")
print("=" * 80)

# =============================================================================
# 1. 多種缺失值處理策略比較
# =============================================================================
print("\n【練習 1】多種缺失值處理策略比較")
print("-" * 80)

df_mean = df_titanic.copy()
df_median = df_titanic.copy()
df_mode = df_titanic.copy()
df_knn = df_titanic.copy()

print("\n1.1 平均值填補策略")
age_mean = df_mean['年齡'].mean()
fare_mean = df_mean['票價'].mean()
df_mean['年齡'].fillna(age_mean, inplace=True)
df_mean['票價'].fillna(fare_mean, inplace=True)
print(f"年齡平均值: {age_mean:.2f}")
print(f"票價平均值: {fare_mean:.2f}")

print("\n1.2 中位數填補策略")
age_median = df_median['年齡'].median()
fare_median = df_median['票價'].median()
df_median['年齡'].fillna(age_median, inplace=True)
df_median['票價'].fillna(fare_median, inplace=True)
print(f"年齡中位數: {age_median:.2f}")
print(f"票價中位數: {fare_median:.2f}")

print("\n1.3 眾數填補策略")
age_mode = df_mode['年齡'].mode()[0]
fare_mode = df_mode['票價'].mode()[0]
df_mode['年齡'].fillna(age_mode, inplace=True)
df_mode['票價'].fillna(fare_mode, inplace=True)
print(f"年齡眾數: {age_mode:.2f}")
print(f"票價眾數: {fare_mode:.2f}")

print("\n1.4 KNN 插補策略（k=5）")
numeric_cols = ['艙等', '年齡', '手足與配偶數', '父母與子女數', '票價']
imputer = KNNImputer(n_neighbors=5)
df_knn[numeric_cols] = imputer.fit_transform(df_knn[numeric_cols])
print("✓ KNN 插補完成（使用 5 個最近鄰居）")

print("\n1.5 不同策略填補後的統計比較（年齡欄位）")
comparison_df = pd.DataFrame({
    '原始資料': df_titanic['年齡'].describe(),
    '平均值填補': df_mean['年齡'].describe(),
    '中位數填補': df_median['年齡'].describe(),
    '眾數填補': df_mode['年齡'].describe(),
    'KNN插補': df_knn['年齡'].describe()
})
display(comparison_df)

print("\n1.6 視覺化不同填補策略的分布")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('不同缺失值處理策略對年齡分布的影響', fontsize=16)

axes[0, 0].hist(df_titanic['年齡'].dropna(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('原始資料（已移除 NA）')
axes[0, 0].set_xlabel('年齡')
axes[0, 0].set_ylabel('頻率')
axes[0, 0].axvline(df_titanic['年齡'].mean(), color='red', linestyle='--', label='平均值')
axes[0, 0].legend()

axes[0, 1].hist(df_mean['年齡'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('平均值填補')
axes[0, 1].set_xlabel('年齡')
axes[0, 1].set_ylabel('頻率')
axes[0, 1].axvline(age_mean, color='red', linestyle='--')

axes[0, 2].hist(df_median['年齡'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 2].set_title('中位數填補')
axes[0, 2].set_xlabel('年齡')
axes[0, 2].set_ylabel('頻率')
axes[0, 2].axvline(age_median, color='red', linestyle='--')

axes[1, 0].hist(df_mode['年齡'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].set_title('眾數填補')
axes[1, 0].set_xlabel('年齡')
axes[1, 0].set_ylabel('頻率')
axes[1, 0].axvline(age_mode, color='red', linestyle='--')

axes[1, 1].hist(df_knn['年齡'], bins=30, edgecolor='black', alpha=0.7, color='brown')
axes[1, 1].set_title('KNN 插補')
axes[1, 1].set_xlabel('年齡')
axes[1, 1].set_ylabel('頻率')
axes[1, 1].axvline(df_knn['年齡'].mean(), color='red', linestyle='--')

data_to_plot = [
    df_titanic['年齡'].dropna(),
    df_mean['年齡'],
    df_median['年齡'],
    df_mode['年齡'],
    df_knn['年齡']
]
axes[1, 2].boxplot(data_to_plot, labels=['原始', '平均', '中位', '眾數', 'KNN'])
axes[1, 2].set_title('箱型圖比較')
axes[1, 2].set_ylabel('年齡')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("✓ 缺失值處理策略比較完成")

# =============================================================================
# 2. 進階特徵工程
# =============================================================================
print("\n【練習 2】進階特徵工程")
print("-" * 80)

df_feature = df_median.copy()

print("\n2.1 特徵組合")

df_feature['家庭人數'] = df_feature['手足與配偶數'] + df_feature['父母與子女數'] + 1
print("✓ 建立特徵：家庭人數 = 手足與配偶數 + 父母與子女數 + 1")

df_feature['獨自一人'] = (df_feature['家庭人數'] == 1).astype(int)
print("✓ 建立特徵：獨自一人 (0=有家人, 1=獨自)")

df_feature['每人平均票價'] = df_feature['票價'] / df_feature['家庭人數']
print("✓ 建立特徵：每人平均票價 = 票價 / 家庭人數")

df_feature['年齡分組'] = pd.cut(df_feature['年齡'],
                               bins=[0, 12, 18, 35, 60, 100],
                               labels=['兒童', '青少年', '青年', '中年', '老年'])
print("✓ 建立特徵：年齡分組（兒童/青少年/青年/中年/老年）")

df_feature['票價等級'] = pd.qcut(df_feature['票價'],
                                q=4,
                                labels=['低', '中低', '中高', '高'])
print("✓ 建立特徵：票價等級（基於四分位數）")

print("\n新建立的特徵統計：")
print(f"家庭人數範圍: {df_feature['家庭人數'].min()} - {df_feature['家庭人數'].max()}")
print(f"獨自一人比例: {df_feature['獨自一人'].mean():.2%}")
print(f"每人平均票價範圍: {df_feature['每人平均票價'].min():.2f} - {df_feature['每人平均票價'].max():.2f}")
print("\n年齡分組分布:")
print(df_feature['年齡分組'].value_counts())
print("\n票價等級分布:")
print(df_feature['票價等級'].value_counts())

print("\n2.2 多項式特徵生成")

poly_features = ['年齡', '票價']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly.fit_transform(df_feature[poly_features].fillna(0))
poly_feature_names = poly.get_feature_names_out(poly_features)

print(f"原始特徵數: {len(poly_features)}")
print(f"多項式特徵數: {len(poly_feature_names)}")
print(f"新增特徵: {poly_feature_names}")

df_poly = pd.DataFrame(poly_data, columns=poly_feature_names)
print("\n多項式特徵範例（前5行）:")
display(df_poly.head())

print("\n2.3 特徵選擇（SelectKBest）")

feature_cols = ['艙等', '年齡', '手足與配偶數', '父母與子女數', '票價',
                '家庭人數', '獨自一人', '每人平均票價']
X = df_feature[feature_cols].fillna(0)
y = df_feature['生還']

selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

print(f"選擇前的特徵數: {len(feature_cols)}")
print(f"選擇後的特徵數: {len(selected_features)}")
print(f"被選中的特徵: {selected_features}")

feature_scores = pd.DataFrame({
    '特徵': feature_cols,
    'F-score': selector.scores_,
    'p-value': selector.pvalues_
}).sort_values('F-score', ascending=False)
print("\n特徵重要性排序:")
display(feature_scores)

print("\n2.4 遞迴特徵消除（RFE）")

df_rfe = df_feature.copy()
df_rfe['性別_編碼'] = df_rfe['性別'].map({'male': 1, 'female': 0})

rfe_features = ['艙等', '性別_編碼', '年齡', '手足與配偶數', '父母與子女數',
                '票價', '家庭人數', '獨自一人']
X_rfe = df_rfe[rfe_features].fillna(0)

estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=5)
rfe.fit(X_rfe, y)

rfe_results = pd.DataFrame({
    '特徵': rfe_features,
    '是否選中': rfe.support_,
    '排名': rfe.ranking_
}).sort_values('排名')
print("\nRFE 特徵選擇結果:")
display(rfe_results)

selected_rfe_features = [feat for feat, selected in zip(rfe_features, rfe.support_) if selected]
print(f"\nRFE 選中的特徵: {selected_rfe_features}")

print("\n2.5 視覺化新特徵與生還的關係")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('新特徵與生還率的關係', fontsize=16)

axes[0, 0].bar(df_feature.groupby('家庭人數')['生還'].mean().index,
               df_feature.groupby('家庭人數')['生還'].mean().values,
               edgecolor='black', alpha=0.7)
axes[0, 0].set_title('家庭人數 vs 生還率')
axes[0, 0].set_xlabel('家庭人數')
axes[0, 0].set_ylabel('生還率')
axes[0, 0].grid(True, alpha=0.3)

alone_survival = df_feature.groupby('獨自一人')['生還'].mean()
axes[0, 1].bar(['有家人', '獨自一人'], alone_survival.values,
               edgecolor='black', alpha=0.7, color=['green', 'orange'])
axes[0, 1].set_title('獨自一人 vs 生還率')
axes[0, 1].set_ylabel('生還率')
axes[0, 1].grid(True, alpha=0.3)

age_group_survival = df_feature.groupby('年齡分組')['生還'].mean()
axes[1, 0].bar(range(len(age_group_survival)), age_group_survival.values,
               tick_label=age_group_survival.index, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('年齡分組 vs 生還率')
axes[1, 0].set_xlabel('年齡分組')
axes[1, 0].set_ylabel('生還率')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

fare_level_survival = df_feature.groupby('票價等級')['生還'].mean()
axes[1, 1].bar(range(len(fare_level_survival)), fare_level_survival.values,
               tick_label=fare_level_survival.index, edgecolor='black', alpha=0.7,
               color='purple')
axes[1, 1].set_title('票價等級 vs 生還率')
axes[1, 1].set_xlabel('票價等級')
axes[1, 1].set_ylabel('生還率')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ 進階特徵工程完成")

# =============================================================================
# 3. 不同標準化方法比較
# =============================================================================
print("\n【練習 3】不同標準化方法比較")
print("-" * 80)

df_scale = df_median.copy()
scale_features = ['年齡', '票價']

print("\n3.1 原始資料統計")
print(df_scale[scale_features].describe())

print("\n3.2 Min-Max 標準化")
minmax_scaler = MinMaxScaler()
df_minmax = df_scale.copy()
df_minmax[scale_features] = minmax_scaler.fit_transform(df_scale[scale_features])
print("Min-Max 標準化後統計（範圍: 0-1）:")
print(df_minmax[scale_features].describe())

print("\n3.3 Standard 標準化（Z-score）")
standard_scaler = StandardScaler()
df_standard = df_scale.copy()
df_standard[scale_features] = standard_scaler.fit_transform(df_scale[scale_features])
print("Standard 標準化後統計（平均=0, 標準差=1）:")
print(df_standard[scale_features].describe())

print("\n3.4 Robust 標準化")
robust_scaler = RobustScaler()
df_robust = df_scale.copy()
df_robust[scale_features] = robust_scaler.fit_transform(df_scale[scale_features])
print("Robust 標準化後統計（使用中位數和 IQR）:")
print(df_robust[scale_features].describe())

print("\n3.5 視覺化不同標準化方法")
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('不同標準化方法比較', fontsize=16)

for idx, feature in enumerate(scale_features):
    axes[idx, 0].hist(df_scale[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[idx, 0].set_title(f'{feature} - 原始資料')
    axes[idx, 0].set_ylabel('頻率')

    axes[idx, 1].hist(df_minmax[feature], bins=30, edgecolor='black',
                      alpha=0.7, color='orange')
    axes[idx, 1].set_title(f'{feature} - Min-Max')

    axes[idx, 2].hist(df_standard[feature], bins=30, edgecolor='black',
                      alpha=0.7, color='green')
    axes[idx, 2].set_title(f'{feature} - Standard')

    axes[idx, 3].hist(df_robust[feature], bins=30, edgecolor='black',
                      alpha=0.7, color='purple')
    axes[idx, 3].set_title(f'{feature} - Robust')

plt.tight_layout()
plt.show()

print("\n3.6 箱型圖比較（票價）")
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('票價不同標準化方法的箱型圖比較', fontsize=16)

axes[0].boxplot(df_scale['票價'])
axes[0].set_title('原始資料')
axes[0].set_ylabel('票價')

axes[1].boxplot(df_minmax['票價'])
axes[1].set_title('Min-Max')
axes[1].set_ylabel('標準化值')

axes[2].boxplot(df_standard['票價'])
axes[2].set_title('Standard')
axes[2].set_ylabel('標準化值')

axes[3].boxplot(df_robust['票價'])
axes[3].set_title('Robust')
axes[3].set_ylabel('標準化值')

plt.tight_layout()
plt.show()

print("\n✓ 標準化方法比較完成")
print("\n標準化方法選擇建議:")
print("  - Min-Max: 適合有明確範圍需求的情況（如神經網路）")
print("  - Standard: 適合資料呈常態分布的情況（最常用）")
print("  - Robust: 適合資料含有異常值的情況")

# =============================================================================
# 4. 資料品質檢測與視覺化
# =============================================================================
print("\n【練習 4】資料品質檢測與視覺化")
print("-" * 80)

df_quality = df_median.copy()

print("\n4.1 異常值檢測（IQR 方法）")

def detect_outliers_iqr(data, column):
    """使用 IQR 方法檢測異常值"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

age_outliers, age_lower, age_upper = detect_outliers_iqr(df_quality, '年齡')
print(f"\n年齡異常值:")
print(f"  下界: {age_lower:.2f}, 上界: {age_upper:.2f}")
print(f"  異常值數量: {len(age_outliers)}")
print(f"  異常值比例: {len(age_outliers) / len(df_quality):.2%}")

fare_outliers, fare_lower, fare_upper = detect_outliers_iqr(df_quality, '票價')
print(f"\n票價異常值:")
print(f"  下界: {fare_lower:.2f}, 上界: {fare_upper:.2f}")
print(f"  異常值數量: {len(fare_outliers)}")
print(f"  異常值比例: {len(fare_outliers) / len(df_quality):.2%}")

print("\n4.2 箱型圖視覺化")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('資料品質檢測：異常值視覺化', fontsize=16)

axes[0, 0].boxplot(df_quality['年齡'])
axes[0, 0].set_title('年齡箱型圖')
axes[0, 0].set_ylabel('年齡')
axes[0, 0].axhline(age_upper, color='red', linestyle='--', label='上界')
axes[0, 0].axhline(age_lower, color='red', linestyle='--', label='下界')
axes[0, 0].legend()

axes[0, 1].boxplot(df_quality['票價'])
axes[0, 1].set_title('票價箱型圖')
axes[0, 1].set_ylabel('票價')
axes[0, 1].axhline(fare_upper, color='red', linestyle='--')
axes[0, 1].axhline(fare_lower, color='red', linestyle='--')

df_quality.boxplot(column='票價', by='艙等', ax=axes[0, 2])
axes[0, 2].set_title('各艙等的票價分布')
axes[0, 2].set_xlabel('艙等')
axes[0, 2].set_ylabel('票價')

axes[1, 0].hist(df_quality['年齡'], bins=40, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('年齡分布直方圖')
axes[1, 0].set_xlabel('年齡')
axes[1, 0].set_ylabel('頻率')
axes[1, 0].axvline(age_lower, color='red', linestyle='--')
axes[1, 0].axvline(age_upper, color='red', linestyle='--')

axes[1, 1].hist(df_quality['票價'], bins=40, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('票價分布直方圖')
axes[1, 1].set_xlabel('票價')
axes[1, 1].set_ylabel('頻率')
axes[1, 1].set_yscale('log')

axes[1, 2].scatter(df_quality['年齡'], df_quality['票價'], alpha=0.5)
axes[1, 2].set_title('年齡 vs 票價散佈圖')
axes[1, 2].set_xlabel('年齡')
axes[1, 2].set_ylabel('票價')
axes[1, 2].axhline(fare_upper, color='red', linestyle='--', alpha=0.5)
axes[1, 2].axvline(age_upper, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\n4.3 相關係數分析")

numeric_df = df_quality[['生還', '艙等', '年齡', '手足與配偶數', '父母與子女數', '票價']]
correlation_matrix = numeric_df.corr()

print("\n相關係數矩陣:")
display(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('數值欄位相關係數熱力圖', fontsize=14)
plt.tight_layout()
plt.show()

print("\n高度相關的特徵對（|相關係數| > 0.5）:")
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            high_corr.append((correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j]))

for feat1, feat2, corr in high_corr:
    print(f"  {feat1} ↔ {feat2}: {corr:.3f}")

print("\n4.4 資料分布檢測（偏態與峰態）")

distribution_stats = pd.DataFrame({
    '欄位': ['年齡', '票價', '手足與配偶數', '父母與子女數'],
    '偏態': [
        stats.skew(df_quality['年齡']),
        stats.skew(df_quality['票價']),
        stats.skew(df_quality['手足與配偶數']),
        stats.skew(df_quality['父母與子女數'])
    ],
    '峰態': [
        stats.kurtosis(df_quality['年齡']),
        stats.kurtosis(df_quality['票價']),
        stats.kurtosis(df_quality['手足與配偶數']),
        stats.kurtosis(df_quality['父母與子女數'])
    ]
})

print("\n偏態與峰態統計:")
print("  偏態 > 0: 右偏（正偏），偏態 < 0: 左偏（負偏）")
print("  峰態 > 0: 尖峰分布，峰態 < 0: 平坦分布")
display(distribution_stats)

print("\n4.5 Q-Q 圖（常態性檢驗）")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Q-Q 圖：檢驗資料常態性', fontsize=16)

features_to_test = ['年齡', '票價', '手足與配偶數', '父母與子女數']

for idx, feature in enumerate(features_to_test):
    ax = axes[idx // 2, idx % 2]
    stats.probplot(df_quality[feature], dist="norm", plot=ax)
    ax.set_title(f'{feature} Q-Q 圖')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n4.6 資料完整性報告")
completeness_report = pd.DataFrame({
    '欄位': df_quality.columns,
    '非空值數量': df_quality.count(),
    '缺失值數量': df_quality.isnull().sum(),
    '完整度 (%)': (df_quality.count() / len(df_quality) * 100)
}).sort_values('完整度 (%)')

print(completeness_report)

print("\n✓ 資料品質檢測與視覺化完成")