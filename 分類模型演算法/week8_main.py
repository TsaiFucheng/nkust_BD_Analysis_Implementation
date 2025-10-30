"""
大數據資料分析實作 - 分類模型演算法練習
Classification Model Algorithms Practice

本腳本實作PDF中提到的所有分類演算法：
1. 邏輯斯迴歸 (Logistic Regression)
2. 支援向量機 (SVM)
3. 神經網路 (Neural Network)
4. 決策樹 (Decision Tree)
5. 隨機森林 (Random Forest)
6. XGBoost
"""

# ============================
# 步驟1: 安裝與設定
# ============================

import warnings
# warnings.filterwarnings('ignore') 我習慣不使用

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


print("=" * 60)
print("大數據資料分析實作 - 分類模型演算法練習")
print("=" * 60)

# ============================
# 步驟2: 建立資料集
# ============================

from sklearn.datasets import make_moons, make_circles, make_classification

print("\n[步驟1] 建立三種不同的資料分布...")

X1, y1 = make_classification(
    n_features=2, 
    n_redundant=0,
    n_informative=2, 
    random_state=random_seed,
    n_clusters_per_class=1, 
    n_samples=200, 
    n_classes=2
)

X2, y2 = make_moons(noise=0.05, random_state=random_seed, n_samples=200)

X3, y3 = make_circles(noise=0.02, random_state=random_seed, n_samples=200)

DataList = [(X1, y1), (X2, y2), (X3, y3)]
N = len(DataList)

print(f"✓ 已建立 {N} 種資料集")
print(f"  - 線性可分離: {X1.shape[0]} 筆")
print(f"  - 新月形: {X2.shape[0]} 筆")
print(f"  - 同心圓: {X3.shape[0]} 筆")

# ============================
# 步驟3: 視覺化原始資料
# ============================

print("\n[步驟2] 繪製原始資料分布...")

from matplotlib.colors import ListedColormap

plt.figure(figsize=(15, 4))
cmap = ListedColormap(['#0000FF', '#000000'])

for i, data in enumerate(DataList):
    X, y = data
    ax = plt.subplot(1, N, i+1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    titles = ['線性可分離', '新月形', '同心圓']
    ax.set_title(titles[i])

plt.tight_layout()
plt.savefig('./outputs/original_data.png', dpi=150, bbox_inches='tight')
print("✓ 原始資料圖已儲存")
plt.close()

# ============================
# 步驟4: 定義決策邊界繪圖函式
# ============================

from sklearn.model_selection import train_test_split

def plot_boundary(ax, X, y, algorithm):
    """繪製分類決策邊界"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=random_seed
    )
    
    cmap1 = plt.cm.bwr
    cmap2 = ListedColormap(['#0000FF', '#000000'])
    
    h = 0.005
    
    algorithm.fit(X_train, y_train)
    
    score_test = algorithm.score(X_test, y_test)
    score_train = algorithm.score(X_train, y_train)
    
    f1_min = X[:, 0].min() - 0.5
    f1_max = X[:, 0].max() + 0.5
    f2_min = X[:, 1].min() - 0.5
    f2_max = X[:, 1].max() + 0.5
    
    f1, f2 = np.meshgrid(
        np.arange(f1_min, f1_max, h),
        np.arange(f2_min, f2_max, h)
    )
    
    if hasattr(algorithm, "decision_function"):
        Z = algorithm.decision_function(np.c_[f1.ravel(), f2.ravel()])
        Z = Z.reshape(f1.shape)
        ax.contour(f1, f2, Z, levels=[0], linewidths=2)
    else:
        Z = algorithm.predict_proba(np.c_[f1.ravel(), f2.ravel()])[:, 1]
        Z = Z.reshape(f1.shape)
    
    ax.contourf(f1, f2, Z, cmap=cmap1, alpha=0.3)
    
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap2)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap2, marker='x')
    
    text = f'測試: {score_test:.2f} 訓練: {score_train:.2f}'
    ax.text(f1.max() - 0.3, f2.min() + 0.3, text, 
            horizontalalignment='right', fontsize=10)

def plot_boundaries(algorithm, DataList, title):
    plt.figure(figsize=(15, 4))
    
    for i, data in enumerate(DataList):
        X, y = data
        ax = plt.subplot(1, N, i+1)
        plot_boundary(ax, X, y, algorithm)
        
        subtitles = ['線性可分離', '新月形', '同心圓']
        ax.set_title(subtitles[i])
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    return plt.gcf()

# ============================
# 步驟5: 測試各種分類演算法
# ============================

print("\n[步驟3] 測試各種分類演算法...")
print("-" * 60)

print("\n 邏輯斯迴歸 (Logistic Regression)")
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(max_iter=1000)
fig = plot_boundaries(model1, DataList, "邏輯斯迴歸 (Logistic Regression)")
plt.savefig('./outputs/logistic_regression.png', dpi=150, bbox_inches='tight')
print("   ✓ 完成 - 線性模型，適合線性可分離資料")
plt.close()

print("\n 支援向量機 (SVM - Linear Kernel)")
from sklearn.svm import SVC

model2 = SVC(kernel='linear', random_state=random_seed)
fig = plot_boundaries(model2, DataList, "支援向量機 - 線性核 (SVM - Linear)")
plt.savefig('./outputs/svm_linear.png', dpi=150, bbox_inches='tight')
print("   ✓ 完成 - 線性核，類似邏輯斯迴歸")
plt.close()

print("\n 支援向量機 (SVM - RBF Kernel)")
model3 = SVC(kernel='rbf', gamma='auto', random_state=random_seed)
fig = plot_boundaries(model3, DataList, "支援向量機 - RBF核 (SVM - RBF)")
plt.savefig('./outputs/svm_rbf.png', dpi=150, bbox_inches='tight')
print("   ✓ 完成 - RBF核，能處理非線性資料")
plt.close()

print("\n神經網路 (Neural Network)")
from sklearn.neural_network import MLPClassifier

model4 = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    random_state=random_seed,
    alpha=0.01
)
fig = plot_boundaries(model4, DataList, "神經網路 (Neural Network)")
plt.savefig('./outputs/neural_network.png', dpi=150, bbox_inches='tight')
print("   ✓ 完成 - 多層感知機，處理複雜非線性")
plt.close()

print("\n5️⃣  決策樹 (Decision Tree)")
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(max_depth=5, random_state=random_seed)
fig = plot_boundaries(model5, DataList, "決策樹 (Decision Tree)")
plt.savefig('./outputs/decision_tree.png', dpi=150, bbox_inches='tight')
print("   ✓ 完成 - 樹狀結構，易於理解")
plt.close()

print("\n6️⃣  隨機森林 (Random Forest)")
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=random_seed
)
fig = plot_boundaries(model6, DataList, "隨機森林 (Random Forest)")
plt.savefig('./outputs/random_forest.png', dpi=150, bbox_inches='tight')
print("   ✓ 完成 - 集成多棵決策樹，準確度高")
plt.close()

print("\n XGBoost (Extreme Gradient Boosting)")
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    
    model7 = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=random_seed,
        eval_metric='logloss'
    )
    fig = plot_boundaries(model7, DataList, "XGBoost (Extreme Gradient Boosting)")
    plt.savefig('./outputs/xgboost.png', dpi=150, bbox_inches='tight')
    print("   ✓ 完成 - 梯度提升，最高準確率")
    plt.close()
except ImportError:
    print("   ⚠ XGBoost 未安裝，跳過此演算法")
    print("   安裝指令: pip install xgboost")

# ============================
# 步驟6: 演算法比較分析
# ============================

print("\n" + "=" * 60)
print("[步驟4] 演算法比較分析")
print("=" * 60)

from sklearn.model_selection import cross_val_score

models = {
    '邏輯斯迴歸': LogisticRegression(max_iter=1000),
    'SVM-Linear': SVC(kernel='linear', random_state=random_seed),
    'SVM-RBF': SVC(kernel='rbf', gamma='auto', random_state=random_seed),
    '神經網路': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                              random_state=random_seed, alpha=0.01),
    '決策樹': DecisionTreeClassifier(max_depth=5, random_state=random_seed),
    '隨機森林': RandomForestClassifier(n_estimators=100, max_depth=5, 
                                      random_state=random_seed)
}

try:
    models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=3, 
                                      random_state=random_seed, eval_metric='logloss')
except:
    pass

results = []
dataset_names = ['線性可分離', '新月形', '同心圓']

for dataset_idx, (X, y) in enumerate(DataList):
    print(f"\n資料集: {dataset_names[dataset_idx]}")
    print("-" * 40)
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        mean_score = scores.mean()
        std_score = scores.std()
        
        results.append({
            '資料集': dataset_names[dataset_idx],
            '演算法': name,
            '平均準確率': mean_score,
            '標準差': std_score
        })
        
        print(f"{name:12s}: {mean_score:.4f} (±{std_score:.4f})")

df_results = pd.DataFrame(results)

df_results.to_csv('./outputs/algorithm_comparison.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 比較結果已儲存至 algorithm_comparison.csv")

plt.figure(figsize=(14, 6))

for idx, dataset_name in enumerate(dataset_names):
    plt.subplot(1, 3, idx+1)
    
    data = df_results[df_results['資料集'] == dataset_name]
    
    plt.barh(data['演算法'], data['平均準確率'], color='skyblue')
    plt.xlabel('準確率')
    plt.title(dataset_name)
    plt.xlim(0, 1.05)
    
    for i, v in enumerate(data['平均準確率']):
        plt.text(v + 0.02, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('./outputs/algorithm_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 比較圖已儲存")
plt.close()