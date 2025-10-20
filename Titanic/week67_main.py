# =============================================================================
# 步驟 1: 載入所需的 Python 模組
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 步驟 2: 設定 numpy、pandas、matplotlib 的顯示選項
# =============================================================================
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
plt.rcParams["font.size"] = 14

# =============================================================================
# 步驟 3: 載入 Titanic 資料集並重新命名欄位
# =============================================================================
print("=" * 80)
print("Titanic 資料集預處理與標準化")
print("=" * 80)

df_titanic = sns.load_dataset("titanic")

columns_t = ['生還', '艙等', '性別', '年齡', '手足與配偶數',
             '父母與子女數', '票價', '乘船港代碼', '艙等名', '男女兒童',
             '成人男子', '甲板', '乘船港', '生還與否', '單身']

df_titanic.columns = columns_t

print(f"\n原始資料維度: {df_titanic.shape}")
print(f"資料筆數: {len(df_titanic)}")
print(f"欄位數量: {len(df_titanic.columns)}")

print("\n=== 原始資料前五行 ===")
display(df_titanic.head())

print("\n=== 各欄位的資料類型 ===")
print(df_titanic.dtypes)

# =============================================================================
# 步驟 4: 刪除多餘欄位
# =============================================================================
print("\n" + "=" * 80)
print("【步驟 1】刪除多餘欄位")
print("=" * 80)

columns_to_drop = ['乘船港代碼', '男女兒童', '成人男子', '生還與否', '單身']

df_processed = df_titanic.drop(columns=columns_to_drop, axis=1)

print(f"刪除欄位: {columns_to_drop}")
print(f"刪除後資料維度: {df_processed.shape}")
print(f"剩餘欄位: {list(df_processed.columns)}")

# =============================================================================
# 步驟 5: 處理缺失值
# =============================================================================
print("\n" + "=" * 80)
print("【步驟 2】處理缺失值")
print("=" * 80)

print("\n5.1 檢查各欄位的缺失值數量")
missing_values = df_processed.isnull().sum()
missing_percentage = (missing_values / len(df_processed)) * 100
missing_df = pd.DataFrame({
    '缺失數量': missing_values,
    '缺失比例 (%)': missing_percentage
})
print(missing_df[missing_df['缺失數量'] > 0].sort_values('缺失數量', ascending=False))

print("\n5.2 處理缺失值")

age_median = df_processed['年齡'].median()
df_processed['年齡'].fillna(age_median, inplace=True)
print(f"✓ 年齡缺失值已用中位數 ({age_median:.2f}) 填補")

embark_mode = df_processed['乘船港'].mode()[0]
df_processed['乘船港'].fillna(embark_mode, inplace=True)
print(f"✓ 乘船港缺失值已用眾數 ({embark_mode}) 填補")


df_processed['甲板'] = df_processed['甲板'].astype(object)
df_processed['甲板'].fillna('Unknown', inplace=True)
print("✓ 甲板缺失值已標記為 'Unknown'")

if df_processed['艙等名'].isnull().any():
    df_processed['艙等名'].fillna(df_processed['艙等名'].mode()[0], inplace=True)
    print("✓ 艙等名缺失值已處理")

print(f"\n處理後剩餘缺失值總數: {df_processed.isnull().sum().sum()}")

# =============================================================================
# 步驟 6: 二元資料轉換（Binary Encoding）
# =============================================================================
print("\n" + "=" * 80)
print("【步驟 3】二元資料轉換")
print("=" * 80)

print("\n6.1 性別欄位轉換")
print("轉換前的值:")
print(df_processed['性別'].value_counts())

df_processed['性別_編碼'] = df_processed['性別'].map({'male': 1, 'female': 0})

print("\n轉換後的值:")
print(df_processed['性別_編碼'].value_counts())
print("✓ 性別已轉換為二元編碼 (male=1, female=0)")

print("\n6.2 生還欄位（已為二元）")
print(df_processed['生還'].value_counts())
print("✓ 生還欄位已為二元格式 (1=生還, 0=死亡)")

# =============================================================================
# 步驟 7: 多類別資料轉換（One-Hot Encoding）
# =============================================================================
print("\n" + "=" * 80)
print("【步驟 4】多類別資料轉換 (One-Hot Encoding)")
print("=" * 80)

print("\n7.1 乘船港 One-Hot Encoding")
print("轉換前的值:")
print(df_processed['乘船港'].value_counts())

embark_dummies = pd.get_dummies(df_processed['乘船港'], prefix='乘船港')
print("\n轉換後的新欄位:")
print(embark_dummies.head())

df_processed = pd.concat([df_processed, embark_dummies], axis=1)
print(f"✓ 乘船港已轉換為 {len(embark_dummies.columns)} 個 One-Hot 欄位")

print("\n7.2 艙等名 One-Hot Encoding")
print("轉換前的值:")
print(df_processed['艙等名'].value_counts())

pclass_dummies = pd.get_dummies(df_processed['艙等名'], prefix='艙等')
print("\n轉換後的新欄位:")
print(pclass_dummies.head())

df_processed = pd.concat([df_processed, pclass_dummies], axis=1)
print(f"✓ 艙等名已轉換為 {len(pclass_dummies.columns)} 個 One-Hot 欄位")

print("\n7.3 甲板 One-Hot Encoding")
print(f"甲板的唯一值數量: {df_processed['甲板'].nunique()}")
print("甲板的值分布:")
print(df_processed['甲板'].value_counts())

deck_dummies = pd.get_dummies(df_processed['甲板'], prefix='甲板')
df_processed = pd.concat([df_processed, deck_dummies], axis=1)
print(f"✓ 甲板已轉換為 {len(deck_dummies.columns)} 個 One-Hot 欄位")

print(f"\n轉換後資料維度: {df_processed.shape}")

# =============================================================================
# 步驟 8: 資料標準化（Standardization）
# =============================================================================
print("\n" + "=" * 80)
print("【步驟 5】資料標準化 (Z-score Standardization)")
print("=" * 80)

numeric_columns = ['年齡', '手足與配偶數', '父母與子女數', '票價']

print("\n8.1 標準化前的統計資訊")
print(df_processed[numeric_columns].describe())

scaler = StandardScaler()
df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

print("\n8.2 標準化後的統計資訊")
print(df_processed[numeric_columns].describe())
print("\n✓ 數值欄位已完成 Z-score 標準化")
print("  標準化後的資料：平均值 ≈ 0，標準差 ≈ 1")

print("\n8.3 繪製標準化後的資料分布")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('標準化後的數值欄位分布', fontsize=16)

for idx, col in enumerate(numeric_columns):
    ax = axes[idx // 2, idx % 2]
    df_processed[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(f'{col} (標準化後)')
    ax.set_xlabel('標準化值')
    ax.set_ylabel('頻率')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='平均值=0')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 步驟 9: 最終資料檢查與匯出
# =============================================================================
print("\n" + "=" * 80)
print("【步驟 6】最終資料檢查")
print("=" * 80)

columns_to_remove = ['性別', '乘船港', '艙等名', '甲板']
df_final = df_processed.drop(columns=columns_to_remove, axis=1)

print(f"\n最終資料維度: {df_final.shape}")
print(f"最終欄位數量: {len(df_final.columns)}")
print("\n最終欄位列表:")
print(list(df_final.columns))

print(f"\n最終資料缺失值總數: {df_final.isnull().sum().sum()}")

print("\n=== 最終處理後的資料前五行 ===")
display(df_final.head())

print("\n=== 最終資料的統計描述 ===")
display(df_final.describe())