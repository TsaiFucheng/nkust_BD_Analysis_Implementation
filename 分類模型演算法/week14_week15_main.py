"""
大數據資料分析實作 - Week 14 & 15 模型評估
Model Evaluation

本腳本實作PDF中提到的模型評估方法：
Part 1: 二元分類評估 (Binary Classification Evaluation)
    - 混淆矩陣 (Confusion Matrix)
    - 精確率、召回率、F1-score (Precision, Recall, F1-score)
    - PR 曲線 (Precision-Recall Curve)
    - ROC 曲線 (Receiver Operating Characteristic Curve)

Part 2: 迴歸評估 (Regression Evaluation)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
"""

# ============================
# 步驟1: 安裝與設定
# ============================

import warnings
# warnings.filterwarnings('ignore') 我習慣不使用

import sys
import io

# 設定 stdout 編碼為 UTF-8（解決 Windows 終端機編碼問題）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 設定隨機種子
random_seed = 123

np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)

import platform
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 14

# 建立輸出資料夾
output_dir = './outputs/week14_15'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 60)
print("大數據資料分析實作 - Week 14 & 15 模型評估")
print("=" * 60)

# ============================
# Part 1: 二元分類評估
# Binary Classification Evaluation
# ============================

print("\n" + "=" * 60)
print("Part 1: 二元分類評估 (Binary Classification)")
print("=" * 60)

# ============================
# 步驟2: 載入資料集
# ============================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

print("\n[步驟1] 載入 Breast Cancer 資料集...")

# 載入乳癌資料集
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

print(f"   資料集大小: {X.shape[0]} 筆資料, {X.shape[1]} 個特徵")
print(f"   目標類別: {target_names}")
print(f"   類別分布: 惡性(0)={sum(y==0)}, 良性(1)={sum(y==1)}")

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed
)

print(f"\n   訓練集大小: {X_train.shape[0]} 筆")
print(f"   測試集大小: {X_test.shape[0]} 筆")

# ============================
# 步驟3: 訓練模型
# ============================

from sklearn.linear_model import LogisticRegression

print("\n[步驟2] 訓練 Logistic Regression 模型...")

# 訓練羅吉斯迴歸模型
model = LogisticRegression(max_iter=10000, random_state=random_seed)
model.fit(X_train, y_train)

# 計算準確率
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"   訓練準確率: {train_accuracy:.4f}")
print(f"   測試準確率: {test_accuracy:.4f}")

# 進行預測
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_proba1 = y_proba[:, 1]  # 類別1（良性）的機率

print(f"\n   預測結果範例 (前5筆):")
print(f"   實際值: {y_test[:5]}")
print(f"   預測值: {y_pred[:5]}")
print(f"   預測機率(類別1): {y_proba1[:5].round(4)}")

# ============================
# 步驟4: 混淆矩陣 (Confusion Matrix)
# ============================

from sklearn.metrics import confusion_matrix

print("\n" + "-" * 50)
print("[步驟3] 混淆矩陣 (Confusion Matrix)")
print("-" * 50)

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)

print("\n混淆矩陣:")
print(cm)

# 定義 make_cm 函式（來自 PDF）
def make_cm(matrix, columns):
    """
    格式化混淆矩陣為 DataFrame

    Parameters:
    -----------
    matrix : array-like
        混淆矩陣
    columns : list
        類別名稱

    Returns:
    --------
    DataFrame : 格式化後的混淆矩陣
    """
    n = len(columns)
    act = ['正確答案數據'] * n
    pred = ['預測結果'] * n
    cm_df = pd.DataFrame(matrix,
                         columns=[pred, columns],
                         index=[act, columns])
    return cm_df

# 使用 make_cm 函式格式化
columns = ['惡性(0)', '良性(1)']
cm_formatted = make_cm(cm, columns)
print("\n格式化混淆矩陣:")
print(cm_formatted)

# 解釋混淆矩陣
TN, FP, FN, TP = cm.ravel()
print(f"\n混淆矩陣分解:")
print(f"   TN (True Negative)  = {TN} (實際惡性, 預測惡性)")
print(f"   FP (False Positive) = {FP} (實際惡性, 預測良性)")
print(f"   FN (False Negative) = {FN} (實際良性, 預測惡性)")
print(f"   TP (True Positive)  = {TP} (實際良性, 預測良性)")

# 手動計算準確率
accuracy_manual = (TP + TN) / (TP + TN + FP + FN)
print(f"\n   手動計算準確率: (TP+TN)/(TP+TN+FP+FN) = {accuracy_manual:.4f}")

# 繪製混淆矩陣熱力圖
plt.figure(figsize=(10, 8))

plt.subplot(1, 1, 1)
im = plt.imshow(cm, cmap='Blues')
plt.colorbar(im)

# 加入數值標籤
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                fontsize=20, color=color)

plt.xticks([0, 1], ['惡性(0)', '良性(1)'])
plt.yticks([0, 1], ['惡性(0)', '良性(1)'])
plt.xlabel('預測結果')
plt.ylabel('正確答案')
plt.title(f'混淆矩陣 (Confusion Matrix)\n準確率 = {test_accuracy:.4f}')

plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"\n   ✓ 圖片已儲存: confusion_matrix.png")
plt.close()

# ============================
# 步驟5: 精確率/召回率/F1-score
# ============================

from sklearn.metrics import precision_recall_fscore_support, classification_report

print("\n" + "-" * 50)
print("[步驟4] 精確率/召回率/F1-score")
print("-" * 50)

# 使用 precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print("\n各類別評估指標:")
print(f"{'類別':<10} {'精確率(Precision)':<20} {'召回率(Recall)':<18} {'F1-score':<12} {'支持度(Support)':<10}")
print("-" * 70)
for i, name in enumerate(['惡性(0)', '良性(1)']):
    print(f"{name:<10} {precision[i]:<20.4f} {recall[i]:<18.4f} {f1[i]:<12.4f} {support[i]:<10}")

# 手動計算精確率和召回率
precision_manual = TP / (TP + FP)
recall_manual = TP / (TP + FN)
f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)

print(f"\n手動計算 (以良性類別為例):")
print(f"   精確率 (Precision) = TP/(TP+FP) = {TP}/({TP}+{FP}) = {precision_manual:.4f}")
print(f"   召回率 (Recall)    = TP/(TP+FN) = {TP}/({TP}+{FN}) = {recall_manual:.4f}")
print(f"   F1-score           = 2*P*R/(P+R) = {f1_manual:.4f}")

# 完整分類報告
print("\n完整分類報告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['惡性(0)', '良性(1)']))

# ============================
# 步驟6: PR 曲線 (Precision-Recall Curve)
# ============================

from sklearn.metrics import precision_recall_curve, auc

print("\n" + "-" * 50)
print("[步驟5] PR 曲線 (Precision-Recall Curve)")
print("-" * 50)

# 計算 PR 曲線
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_proba1)

# 計算 PR-AUC
pr_auc = auc(recall_curve, precision_curve)

print(f"\n   PR-AUC: {pr_auc:.4f}")
print(f"   閾值數量: {len(thresholds_pr)}")

# 繪製 PR 曲線
plt.figure(figsize=(10, 8))

plt.plot(recall_curve, precision_curve, 'b-', linewidth=2,
         label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.fill_between(recall_curve, precision_curve, alpha=0.2)

# 標記不同閾值點
threshold_points = [0.3, 0.5, 0.7, 0.9]
for thresh in threshold_points:
    idx = np.argmin(np.abs(thresholds_pr - thresh))
    if idx < len(precision_curve) - 1:
        plt.scatter(recall_curve[idx], precision_curve[idx], s=100, zorder=5)
        plt.annotate(f'閾值={thresh}',
                    (recall_curve[idx], precision_curve[idx]),
                    textcoords="offset points",
                    xytext=(10, -10),
                    fontsize=10)

plt.xlabel('召回率 (Recall)')
plt.ylabel('精確率 (Precision)')
plt.title(f'PR 曲線 (Precision-Recall Curve)\nAUC = {pr_auc:.4f}')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])

plt.tight_layout()
plt.savefig(f'{output_dir}/pr_curve.png', dpi=150, bbox_inches='tight')
print(f"   ✓ 圖片已儲存: pr_curve.png")
plt.close()

# ============================
# 步驟7: ROC 曲線 (Receiver Operating Characteristic)
# ============================

from sklearn.metrics import roc_curve, roc_auc_score

print("\n" + "-" * 50)
print("[步驟6] ROC 曲線 (Receiver Operating Characteristic)")
print("-" * 50)

# 計算 ROC 曲線
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba1)

# 計算 ROC-AUC
roc_auc = auc(fpr, tpr)
roc_auc_score_val = roc_auc_score(y_test, y_proba1)

print(f"\n   ROC-AUC (auc函式): {roc_auc:.4f}")
print(f"   ROC-AUC (roc_auc_score): {roc_auc_score_val:.4f}")
print(f"   閾值數量: {len(thresholds_roc)}")

# 繪製 ROC 曲線
plt.figure(figsize=(10, 8))

plt.plot(fpr, tpr, 'b-', linewidth=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier (AUC = 0.5)')
plt.fill_between(fpr, tpr, alpha=0.2)

# 標記不同閾值點
for thresh in threshold_points:
    idx = np.argmin(np.abs(thresholds_roc - thresh))
    if idx < len(fpr):
        plt.scatter(fpr[idx], tpr[idx], s=100, zorder=5)
        plt.annotate(f'閾值={thresh}',
                    (fpr[idx], tpr[idx]),
                    textcoords="offset points",
                    xytext=(10, -10),
                    fontsize=10)

plt.xlabel('假陽性率 (False Positive Rate)')
plt.ylabel('真陽性率 (True Positive Rate)')
plt.title(f'ROC 曲線 (Receiver Operating Characteristic)\nAUC = {roc_auc:.4f}')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])

plt.tight_layout()
plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
print(f"   ✓ 圖片已儲存: roc_curve.png")
plt.close()

# 繪製 PR 曲線和 ROC 曲線並排比較
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(recall_curve, precision_curve, 'b-', linewidth=2,
         label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.fill_between(recall_curve, precision_curve, alpha=0.2)
plt.xlabel('召回率 (Recall)')
plt.ylabel('精確率 (Precision)')
plt.title('PR 曲線')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, 'b-', linewidth=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('假陽性率 (FPR)')
plt.ylabel('真陽性率 (TPR)')
plt.title('ROC 曲線')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])

plt.suptitle('PR 曲線與 ROC 曲線比較', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/pr_roc_comparison.png', dpi=150, bbox_inches='tight')
print(f"   ✓ 圖片已儲存: pr_roc_comparison.png")
plt.close()

# ============================
# 二元分類評估總結
# ============================

print("\n" + "=" * 60)
print("Part 1 總結: 二元分類評估結果")
print("=" * 60)

classification_summary = {
    '指標': ['準確率 (Accuracy)', '精確率 (Precision)', '召回率 (Recall)',
             'F1-score', 'PR-AUC', 'ROC-AUC'],
    '數值': [test_accuracy, precision[1], recall[1], f1[1], pr_auc, roc_auc]
}

df_classification = pd.DataFrame(classification_summary)
print("\n")
print(df_classification.to_string(index=False))

# ============================
# Part 2: 迴歸評估
# Regression Evaluation
# ============================

print("\n" + "=" * 60)
print("Part 2: 迴歸評估 (Regression Evaluation)")
print("=" * 60)

# ============================
# 步驟8: 載入迴歸資料集
# ============================

from sklearn.datasets import fetch_california_housing

print("\n[步驟7] 載入 California Housing 資料集...")

# 載入加州房價資料集
housing = fetch_california_housing()
X_reg = housing.data
y_reg = housing.target
feature_names_reg = housing.feature_names

print(f"   資料集大小: {X_reg.shape[0]} 筆資料, {X_reg.shape[1]} 個特徵")
print(f"   特徵名稱: {feature_names_reg}")
print(f"   目標值範圍: {y_reg.min():.2f} ~ {y_reg.max():.2f} (單位: $100,000)")

# 分割訓練集與測試集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=random_seed
)

print(f"\n   訓練集大小: {X_train_reg.shape[0]} 筆")
print(f"   測試集大小: {X_test_reg.shape[0]} 筆")

# ============================
# 步驟9: 訓練迴歸模型
# ============================

from sklearn.linear_model import LinearRegression

print("\n[步驟8] 訓練 Linear Regression 模型...")

# 訓練線性迴歸模型
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# 進行預測
y_pred_reg = reg_model.predict(X_test_reg)

print(f"   模型已訓練完成")
print(f"\n   預測結果範例 (前5筆):")
print(f"   實際值: {y_test_reg[:5].round(4)}")
print(f"   預測值: {y_pred_reg[:5].round(4)}")

# ============================
# 步驟10: 迴歸評估指標
# ============================

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n" + "-" * 50)
print("[步驟9] 迴歸評估指標")
print("-" * 50)

# 計算評估指標
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\n迴歸評估指標:")
print(f"   MSE  (Mean Squared Error)        = {mse:.4f}")
print(f"   RMSE (Root Mean Squared Error)   = {rmse:.4f}")
print(f"   MAE  (Mean Absolute Error)       = {mae:.4f}")
print(f"   R²   (Coefficient of Determination) = {r2:.4f}")

# 手動計算示範
print(f"\n評估指標公式說明:")
print(f"   MSE  = (1/n) * Σ(yi - ŷi)²")
print(f"   RMSE = √MSE")
print(f"   MAE  = (1/n) * Σ|yi - ŷi|")
print(f"   R²   = 1 - (SS_res / SS_tot)")

# 計算殘差
residuals = y_test_reg - y_pred_reg

print(f"\n殘差統計:")
print(f"   殘差平均值: {residuals.mean():.6f}")
print(f"   殘差標準差: {residuals.std():.4f}")
print(f"   殘差最小值: {residuals.min():.4f}")
print(f"   殘差最大值: {residuals.max():.4f}")

# ============================
# 步驟11: 迴歸視覺化
# ============================

print("\n" + "-" * 50)
print("[步驟10] 迴歸視覺化")
print("-" * 50)

# 繪製預測值 vs 實際值散布圖
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, s=20)
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         'r--', linewidth=2, label='完美預測線')
plt.xlabel('實際值 (Actual)')
plt.ylabel('預測值 (Predicted)')
plt.title(f'預測值 vs 實際值\nR² = {r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 繪製殘差圖
plt.subplot(1, 2, 2)
plt.scatter(y_pred_reg, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('預測值 (Predicted)')
plt.ylabel('殘差 (Residual)')
plt.title(f'殘差圖\nMAE = {mae:.4f}, RMSE = {rmse:.4f}')
plt.grid(True, alpha=0.3)

plt.suptitle('California Housing 迴歸分析', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/regression_scatter.png', dpi=150, bbox_inches='tight')
print(f"   ✓ 圖片已儲存: regression_scatter.png")
plt.close()

# 繪製殘差分布直方圖
plt.figure(figsize=(10, 6))

plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零殘差線')
plt.axvline(x=residuals.mean(), color='g', linestyle='-', linewidth=2,
            label=f'殘差平均值 ({residuals.mean():.4f})')
plt.xlabel('殘差值')
plt.ylabel('頻率')
plt.title('殘差分布直方圖')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/residual_histogram.png', dpi=150, bbox_inches='tight')
print(f"   ✓ 圖片已儲存: residual_histogram.png")
plt.close()

# ============================
# 迴歸評估總結
# ============================

print("\n" + "=" * 60)
print("Part 2 總結: 迴歸評估結果")
print("=" * 60)

regression_summary = {
    '指標': ['MSE', 'RMSE', 'MAE', 'R²'],
    '數值': [mse, rmse, mae, r2],
    '說明': ['均方誤差', '均方根誤差', '平均絕對誤差', '決定係數']
}

df_regression = pd.DataFrame(regression_summary)
print("\n")
print(df_regression.to_string(index=False))

# ============================
# 儲存評估結果
# ============================

print("\n" + "=" * 60)
print("儲存評估結果")
print("=" * 60)

# 合併所有評估結果
all_results = []

# 分類評估結果
for i, (metric, value) in enumerate(zip(classification_summary['指標'],
                                         classification_summary['數值'])):
    all_results.append({
        '類型': '分類評估',
        '指標': metric,
        '數值': value
    })

# 迴歸評估結果
for i, (metric, value, desc) in enumerate(zip(regression_summary['指標'],
                                               regression_summary['數值'],
                                               regression_summary['說明'])):
    all_results.append({
        '類型': '迴歸評估',
        '指標': f'{metric} ({desc})',
        '數值': value
    })

df_all_results = pd.DataFrame(all_results)
df_all_results.to_csv(f'{output_dir}/evaluation_summary.csv', index=False, encoding='utf-8-sig')
print(f"   ✓ 評估結果已儲存: evaluation_summary.csv")

# 儲存混淆矩陣
cm_df = pd.DataFrame(cm,
                     columns=['預測_惡性(0)', '預測_良性(1)'],
                     index=['實際_惡性(0)', '實際_良性(1)'])
cm_df.to_csv(f'{output_dir}/confusion_matrix.csv', encoding='utf-8-sig')
print(f"   ✓ 混淆矩陣已儲存: confusion_matrix.csv")

# ============================
# 大數據分析流程總結
# ============================

print("\n" + "=" * 60)
print("大數據分析八步驟流程總結")
print("=" * 60)

steps = """
1. 定義問題 (Define Problem)
   - 明確分析目標：分類？迴歸？聚類？

2. 蒐集資料 (Collect Data)
   - 資料來源：資料庫、API、CSV、Web Scraping

3. 資料探索 (Explore Data)
   - 統計描述、視覺化、缺失值檢查

4. 資料預處理 (Preprocess Data)
   - 處理缺失值、編碼分類變數、標準化/正規化

5. 特徵工程 (Feature Engineering)
   - 特徵選取、特徵轉換、降維

6. 建立模型 (Build Model)
   - 選擇演算法、訓練模型

7. 評估模型 (Evaluate Model)  ← Week 14 & 15 主題
   - 分類：Accuracy, Precision, Recall, F1, ROC-AUC
   - 迴歸：MSE, RMSE, MAE, R²

8. 部署應用 (Deploy Model)
   - 模型儲存、API服務、監控維護
"""

print(steps)

# ============================
# 完成
# ============================

print("\n" + "=" * 60)
print("程式執行完成！")
print("=" * 60)

print(f"\n所有輸出檔案已儲存至: {output_dir}/")
print("\n生成的檔案清單:")
import glob
output_files = glob.glob(f'{output_dir}/*')
for f in sorted(output_files):
    print(f"   - {os.path.basename(f)}")

print("\n" + "=" * 60)
print("Week 14 & 15 模型評估練習完成！")
print("=" * 60)
