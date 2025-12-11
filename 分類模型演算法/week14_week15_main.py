"""
大數據資料分析實作 - Week 14 & 15 期末報告
Binary Classification Model Evaluation - Breast Cancer Dataset

本程式實作課程 PDF 中的 8 步驟大數據分析流程：
1. 載入資料 (Load Data)
2. 了解資料 (Understand Data)
3. 資料預處理 (Preprocess Data)
4. 分割資料 (Split Data)
5. 選擇演算法 (Select Algorithm)
6. 訓練模型 (Train Model)
7. 預測結果 (Predict)
8. 評估模型 (Evaluate Model)

評估指標包含：
- 準確率 (Accuracy)
- 混淆矩陣 (Confusion Matrix)
- 精確率 (Precision)
- 召回率 (Recall)
- F1-score
- ROC 曲線與 AUC
- PR 曲線
"""

# ============================
# 環境設定
# ============================

import warnings
import sys
import io
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 中文編碼支持
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 設定隨機種子
random_seed = 123
np.random.seed(random_seed)

# 數值顯示設定
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)

# 跨平台中文字型設定
import platform
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 12

# 建立輸出資料夾
output_dir = './outputs/week14_15'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已建立輸出資料夾: {output_dir}")

print("=" * 70)
print("大數據資料分析實作 - Week 14 & 15 期末報告")
print("二元分類模型評估 - 乳癌資料集 (Breast Cancer Wisconsin)")
print("=" * 70)

# ============================
# 步驟 1: 載入資料 (Load Data)
# ============================

print("\n" + "=" * 70)
print("步驟 1: 載入資料 (Load Data)")
print("=" * 70)

from sklearn.datasets import load_breast_cancer

# 載入乳癌資料集
cancer = load_breast_cancer()

print(f"\n資料集名稱: {cancer.filename if hasattr(cancer, 'filename') else 'Breast Cancer Wisconsin'}")
print(f"樣本數量: {cancer.data.shape[0]}")
print(f"特徵數量: {cancer.data.shape[1]}")
print(f"類別名稱: {cancer.target_names}")
print(f"  - 0: {cancer.target_names[0]} (惡性)")
print(f"  - 1: {cancer.target_names[1]} (良性)")

# ============================
# 步驟 2: 了解資料 (Understand Data)
# ============================

print("\n" + "=" * 70)
print("步驟 2: 了解資料 (Understand Data)")
print("=" * 70)

# 建立 DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

print("\n--- 資料集前 5 筆 ---")
print(df.head())

print("\n--- 資料統計摘要 ---")
print(df.describe())

print("\n--- 類別分布 ---")
class_counts = df['diagnosis'].value_counts()
print(class_counts)
print(f"\n惡性 (malignant): {class_counts['malignant']} 筆 ({class_counts['malignant']/len(df)*100:.1f}%)")
print(f"良性 (benign): {class_counts['benign']} 筆 ({class_counts['benign']/len(df)*100:.1f}%)")

# 視覺化類別分布
plt.figure(figsize=(8, 5))
colors = ['#ff6b6b', '#4ecdc4']
bars = plt.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
plt.title('乳癌資料集類別分布', fontsize=14, fontweight='bold')
plt.xlabel('診斷結果')
plt.ylabel('樣本數量')
for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n已儲存: {output_dir}/class_distribution.png")

# ============================
# 步驟 3: 資料預處理 (Preprocess Data)
# ============================

print("\n" + "=" * 70)
print("步驟 3: 資料預處理 (Preprocess Data)")
print("=" * 70)

# 檢查缺失值
print("\n--- 檢查缺失值 ---")
missing = df.isnull().sum().sum()
print(f"缺失值總數: {missing}")

# 準備特徵與標籤
X = cancer.data
y = cancer.target

print(f"\n特徵矩陣 X 形狀: {X.shape}")
print(f"標籤向量 y 形狀: {y.shape}")

# 特徵標準化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- 特徵標準化 (StandardScaler) ---")
print(f"標準化前 - 平均值範圍: [{X.mean(axis=0).min():.2f}, {X.mean(axis=0).max():.2f}]")
print(f"標準化後 - 平均值範圍: [{X_scaled.mean(axis=0).min():.4f}, {X_scaled.mean(axis=0).max():.4f}]")
print(f"標準化後 - 標準差範圍: [{X_scaled.std(axis=0).min():.4f}, {X_scaled.std(axis=0).max():.4f}]")

# ============================
# 步驟 4: 分割資料 (Split Data)
# ============================

print("\n" + "=" * 70)
print("步驟 4: 分割資料 (Split Data)")
print("=" * 70)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=random_seed,
    stratify=y  # 保持類別比例
)

print(f"\n訓練集大小: {X_train.shape[0]} 筆 ({X_train.shape[0]/len(y)*100:.0f}%)")
print(f"測試集大小: {X_test.shape[0]} 筆 ({X_test.shape[0]/len(y)*100:.0f}%)")
print(f"\n訓練集類別分布:")
print(f"  - 惡性 (0): {(y_train == 0).sum()} 筆")
print(f"  - 良性 (1): {(y_train == 1).sum()} 筆")
print(f"\n測試集類別分布:")
print(f"  - 惡性 (0): {(y_test == 0).sum()} 筆")
print(f"  - 良性 (1): {(y_test == 1).sum()} 筆")

# ============================
# 步驟 5: 選擇演算法 (Select Algorithm)
# ============================

print("\n" + "=" * 70)
print("步驟 5: 選擇演算法 (Select Algorithm)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression

print("\n選擇演算法: 邏輯斯迴歸 (Logistic Regression)")
print("\n選擇原因:")
print("  1. 適用於二元分類問題")
print("  2. 輸出機率值，方便調整分類閾值")
print("  3. 模型可解釋性高")
print("  4. 訓練速度快，適合中等規模資料集")

model = LogisticRegression(
    random_state=random_seed,
    max_iter=1000,
    solver='lbfgs'
)

print(f"\n模型參數:")
print(f"  - solver: {model.solver}")
print(f"  - max_iter: {model.max_iter}")
print(f"  - random_state: {model.random_state}")

# ============================
# 步驟 6: 訓練模型 (Train Model)
# ============================

print("\n" + "=" * 70)
print("步驟 6: 訓練模型 (Train Model)")
print("=" * 70)

model.fit(X_train, y_train)

print("\n模型訓練完成!")
print(f"模型截距 (intercept): {model.intercept_[0]:.4f}")
print(f"模型係數數量: {len(model.coef_[0])}")

# 顯示前 5 個最重要的特徵
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\n前 5 個最重要的特徵 (依係數絕對值排序):")
print(feature_importance.head())

# ============================
# 步驟 7: 預測結果 (Predict)
# ============================

print("\n" + "=" * 70)
print("步驟 7: 預測結果 (Predict)")
print("=" * 70)

# 預測類別
y_pred = model.predict(X_test)

# 預測機率
y_pred_proba = model.predict_proba(X_test)

print(f"\n預測完成!")
print(f"預測類別形狀: {y_pred.shape}")
print(f"預測機率形狀: {y_pred_proba.shape}")

print("\n--- 預測結果範例 (前 10 筆) ---")
pred_df = pd.DataFrame({
    '實際值': y_test[:10],
    '預測值': y_pred[:10],
    '惡性機率': y_pred_proba[:10, 0],
    '良性機率': y_pred_proba[:10, 1]
})
print(pred_df)

# ============================
# 步驟 8: 評估模型 (Evaluate Model)
# ============================

print("\n" + "=" * 70)
print("步驟 8: 評估模型 (Evaluate Model)")
print("=" * 70)

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# -------------------------
# 8.1 準確率 (Accuracy)
# -------------------------
print("\n" + "-" * 50)
print("8.1 準確率 (Accuracy)")
print("-" * 50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n準確率 (Accuracy) = {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n公式: Accuracy = (TP + TN) / (TP + TN + FP + FN)")

# -------------------------
# 8.2 混淆矩陣 (Confusion Matrix)
# -------------------------
print("\n" + "-" * 50)
print("8.2 混淆矩陣 (Confusion Matrix)")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n混淆矩陣:")
print(f"              預測")
print(f"              惡性    良性")
print(f"實際 惡性    {tn:4d}    {fp:4d}")
print(f"     良性    {fn:4d}    {tp:4d}")

print(f"\n詳細說明:")
print(f"  - True Negative (TN): {tn} - 實際惡性，預測惡性")
print(f"  - False Positive (FP): {fp} - 實際惡性，預測良性 (Type I Error)")
print(f"  - False Negative (FN): {fn} - 實際良性，預測惡性 (Type II Error)")
print(f"  - True Positive (TP): {tp} - 實際良性，預測良性")

# 儲存混淆矩陣 CSV
cm_df = pd.DataFrame(cm,
                     index=['實際: 惡性', '實際: 良性'],
                     columns=['預測: 惡性', '預測: 良性'])
cm_df.to_csv(f'{output_dir}/confusion_matrix.csv', encoding='utf-8-sig')
print(f"\n已儲存: {output_dir}/confusion_matrix.csv")

# 繪製混淆矩陣熱力圖
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['惡性 (0)', '良性 (1)'],
            yticklabels=['惡性 (0)', '良性 (1)'],
            annot_kws={'size': 16})
plt.title('混淆矩陣 (Confusion Matrix)', fontsize=14, fontweight='bold')
plt.xlabel('預測值', fontsize=12)
plt.ylabel('實際值', fontsize=12)

# 添加 TP, TN, FP, FN 標籤
plt.text(0.5, 0.3, f'TN={tn}', ha='center', va='center', fontsize=10, color='gray')
plt.text(1.5, 0.3, f'FP={fp}', ha='center', va='center', fontsize=10, color='gray')
plt.text(0.5, 1.3, f'FN={fn}', ha='center', va='center', fontsize=10, color='gray')
plt.text(1.5, 1.3, f'TP={tp}', ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已儲存: {output_dir}/confusion_matrix.png")

# -------------------------
# 8.3 精確率 (Precision)
# -------------------------
print("\n" + "-" * 50)
print("8.3 精確率 (Precision)")
print("-" * 50)

precision = precision_score(y_test, y_pred)
print(f"\n精確率 (Precision) = {precision:.4f} ({precision*100:.2f}%)")
print(f"\n公式: Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {tp/(tp+fp):.4f}")
print(f"\n意義: 預測為良性的樣本中，實際為良性的比例")

# -------------------------
# 8.4 召回率 (Recall)
# -------------------------
print("\n" + "-" * 50)
print("8.4 召回率 (Recall / Sensitivity)")
print("-" * 50)

recall = recall_score(y_test, y_pred)
print(f"\n召回率 (Recall) = {recall:.4f} ({recall*100:.2f}%)")
print(f"\n公式: Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {tp/(tp+fn):.4f}")
print(f"\n意義: 實際為良性的樣本中，被正確預測為良性的比例")

# -------------------------
# 8.5 F1-score
# -------------------------
print("\n" + "-" * 50)
print("8.5 F1-score")
print("-" * 50)

f1 = f1_score(y_test, y_pred)
print(f"\nF1-score = {f1:.4f}")
print(f"\n公式: F1 = 2 * (Precision * Recall) / (Precision + Recall)")
print(f"     F1 = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
print(f"     F1 = {2 * precision * recall / (precision + recall):.4f}")
print(f"\n意義: 精確率與召回率的調和平均數")

# -------------------------
# 8.6 分類報告 (Classification Report)
# -------------------------
print("\n" + "-" * 50)
print("8.6 分類報告 (Classification Report)")
print("-" * 50)

print("\n完整分類報告:")
report = classification_report(y_test, y_pred,
                               target_names=['惡性 (malignant)', '良性 (benign)'])
print(report)

# -------------------------
# 8.7 ROC 曲線與 AUC
# -------------------------
print("\n" + "-" * 50)
print("8.7 ROC 曲線與 AUC")
print("-" * 50)

# 計算 ROC 曲線
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

print(f"\nAUC (Area Under ROC Curve) = {roc_auc:.4f}")
print(f"\nROC 曲線說明:")
print(f"  - 橫軸: False Positive Rate (FPR) = FP / (FP + TN)")
print(f"  - 縱軸: True Positive Rate (TPR) = TP / (TP + FN) = Recall")
print(f"  - AUC 越接近 1 表示模型效能越好")

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#e74c3c', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
         label='Random Classifier (AUC = 0.5)')
plt.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC 曲線 (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n已儲存: {output_dir}/roc_curve.png")

# -------------------------
# 8.8 PR 曲線
# -------------------------
print("\n" + "-" * 50)
print("8.8 PR 曲線 (Precision-Recall Curve)")
print("-" * 50)

# 計算 PR 曲線
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba[:, 1])
ap = average_precision_score(y_test, y_pred_proba[:, 1])

print(f"\nAverage Precision (AP) = {ap:.4f}")
print(f"\nPR 曲線說明:")
print(f"  - 橫軸: Recall (召回率)")
print(f"  - 縱軸: Precision (精確率)")
print(f"  - AP 是 PR 曲線下的面積，越接近 1 越好")

# 繪製 PR 曲線
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='#3498db', lw=2,
         label=f'PR Curve (AP = {ap:.4f})')
plt.fill_between(recall_curve, precision_curve, alpha=0.3, color='#3498db')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (召回率)', fontsize=12)
plt.ylabel('Precision (精確率)', fontsize=12)
plt.title('PR 曲線 (Precision-Recall Curve)', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/pr_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n已儲存: {output_dir}/pr_curve.png")

# -------------------------
# 8.9 綜合比較圖
# -------------------------
print("\n" + "-" * 50)
print("8.9 ROC 與 PR 曲線比較")
print("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC 曲線
axes[0].plot(fpr, tpr, color='#e74c3c', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
axes[0].fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].set_title('ROC 曲線', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(True, alpha=0.3)

# PR 曲線
axes[1].plot(recall_curve, precision_curve, color='#3498db', lw=2,
             label=f'PR Curve (AP = {ap:.4f})')
axes[1].fill_between(recall_curve, precision_curve, alpha=0.3, color='#3498db')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('PR 曲線', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/pr_roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n已儲存: {output_dir}/pr_roc_comparison.png")

# -------------------------
# 特徵重要性圖
# -------------------------
print("\n" + "-" * 50)
print("特徵重要性分析")
print("-" * 50)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features['coefficient']]
bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('係數值 (Coefficient)', fontsize=12)
plt.ylabel('特徵名稱', fontsize=12)
plt.title('邏輯斯迴歸特徵重要性 (前 15 名)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n已儲存: {output_dir}/feature_importance.png")

# ============================
# 結果總結
# ============================

print("\n" + "=" * 70)
print("模型評估結果總結")
print("=" * 70)

summary_data = {
    '評估指標': ['準確率 (Accuracy)', '精確率 (Precision)', '召回率 (Recall)',
                'F1-score', 'ROC AUC', 'Average Precision'],
    '數值': [accuracy, precision, recall, f1, roc_auc, ap],
    '百分比': [f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', f'{recall*100:.2f}%',
              f'{f1*100:.2f}%', f'{roc_auc*100:.2f}%', f'{ap*100:.2f}%']
}

summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))

# 儲存評估結果
summary_df.to_csv(f'{output_dir}/evaluation_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n已儲存: {output_dir}/evaluation_summary.csv")

print("\n" + "=" * 70)
print("所有輸出檔案")
print("=" * 70)
print(f"""
1. {output_dir}/class_distribution.png    - 類別分布圖
2. {output_dir}/confusion_matrix.png      - 混淆矩陣熱力圖
3. {output_dir}/confusion_matrix.csv      - 混淆矩陣數據
4. {output_dir}/roc_curve.png             - ROC 曲線圖
5. {output_dir}/pr_curve.png              - PR 曲線圖
6. {output_dir}/pr_roc_comparison.png     - ROC 與 PR 比較圖
7. {output_dir}/feature_importance.png    - 特徵重要性圖
8. {output_dir}/evaluation_summary.csv    - 評估指標摘要
""")

print("=" * 70)
print("程式執行完成!")
print("=" * 70)
