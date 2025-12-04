"""
大數據資料分析實作 - Week 11 & 12 樹狀分類演算法
Tree-based Classification Algorithms

本腳本實作PDF中提到的樹狀分類演算法：
1. 決策樹 (Decision Tree) - 含 Iris 資料集視覺化
2. 隨機森林 (Random Forest) - 含參數調整
3. XGBoost (Extreme Gradient Boosting) - 含 GridSearchCV
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
output_dir = './outputs/week11'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 60)
print("大數據資料分析實作 - Week 11 & 12 樹狀分類演算法")
print("=" * 60)

# ============================
# 步驟2: 建立資料集
# ============================

from sklearn.datasets import make_moons, make_circles, make_classification, load_iris

print("\n[步驟1] 建立資料集...")

# 建立三種不同分布的二元分類資料
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
dataset_names = ['線性可分離', '新月形', '同心圓']
N = len(DataList)

print(f"✓ 已建立 {N} 種二元分類資料集")
print(f"  - 線性可分離: {X1.shape[0]} 筆")
print(f"  - 新月形: {X2.shape[0]} 筆")
print(f"  - 同心圓: {X3.shape[0]} 筆")

# 載入 Iris 資料集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
print(f"✓ 已載入 Iris 資料集: {X_iris.shape[0]} 筆, {X_iris.shape[1]} 個特徵")

# ============================
# 步驟3: 定義繪圖函式
# ============================

from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

def plot_boundary(ax, X, y, algorithm, random_seed=123):
    """繪製分類決策邊界"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=random_seed
    )

    cmap1 = plt.cm.bwr
    cmap2 = ListedColormap(['#0000FF', '#000000'])

    h = 0.02

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

    return score_train, score_test

def plot_boundaries(algorithm, DataList, title, filename=None):
    """繪製多個資料集的決策邊界"""
    plt.figure(figsize=(15, 4))

    for i, data in enumerate(DataList):
        X, y = data
        ax = plt.subplot(1, N, i+1)
        plot_boundary(ax, X, y, algorithm)
        ax.set_title(dataset_names[i])

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if filename:
        plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
        print(f"   ✓ 圖片已儲存: {filename}")

    plt.close()

# ============================
# 步驟4: 決策樹 (Decision Tree)
# ============================

print("\n" + "=" * 60)
print("[步驟2] 決策樹 (Decision Tree)")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier, export_graphviz

# 4.1 基本決策樹 - 使用 Iris 資料集
print("\n[2.1] 使用 Iris 資料集訓練決策樹...")

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=random_seed
)

dt_iris = DecisionTreeClassifier(max_depth=3, random_state=random_seed)
dt_iris.fit(X_train_iris, y_train_iris)

train_acc = dt_iris.score(X_train_iris, y_train_iris)
test_acc = dt_iris.score(X_test_iris, y_test_iris)
print(f"   訓練準確率: {train_acc:.4f}")
print(f"   測試準確率: {test_acc:.4f}")

# 嘗試使用 graphviz 視覺化決策樹
print("\n[2.2] 視覺化決策樹...")
try:
    import graphviz
    from sklearn.tree import export_graphviz

    dot_data = export_graphviz(
        dt_iris,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render(f'{output_dir}/decision_tree_iris', format='png', cleanup=True)
    print("   ✓ 決策樹視覺化已儲存: decision_tree_iris.png")

except ImportError:
    print("   ⚠ graphviz 未安裝，使用 matplotlib 繪製決策樹...")
    from sklearn.tree import plot_tree

    plt.figure(figsize=(20, 10))
    plot_tree(dt_iris,
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('決策樹 - Iris 資料集', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decision_tree_iris.png', dpi=150, bbox_inches='tight')
    print("   ✓ 決策樹視覺化已儲存: decision_tree_iris.png")
    plt.close()

# 4.2 max_depth 參數實驗
print("\n[2.3] max_depth 參數實驗...")

max_depths = [1, 2, 3, 5, 10, None]
depth_results = []

# 使用第一個資料集（線性可分離）進行實驗
X, y = DataList[0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed
)

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=random_seed)
    dt.fit(X_train, y_train)

    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)

    depth_label = str(depth) if depth else 'None'
    depth_results.append({
        'max_depth': depth_label,
        '訓練準確率': train_acc,
        '測試準確率': test_acc
    })
    print(f"   max_depth={depth_label:4s}: 訓練={train_acc:.4f}, 測試={test_acc:.4f}")

# 繪製 max_depth 效果圖
plt.figure(figsize=(12, 5))

# 子圖1: 準確率比較
plt.subplot(1, 2, 1)
df_depth = pd.DataFrame(depth_results)
x_pos = range(len(max_depths))
width = 0.35

plt.bar([p - width/2 for p in x_pos], df_depth['訓練準確率'], width, label='訓練', color='skyblue')
plt.bar([p + width/2 for p in x_pos], df_depth['測試準確率'], width, label='測試', color='salmon')
plt.xticks(x_pos, df_depth['max_depth'])
plt.xlabel('max_depth')
plt.ylabel('準確率')
plt.title('決策樹 max_depth 對準確率的影響')
plt.legend()
plt.ylim(0, 1.1)

# 子圖2: 不同深度的決策邊界
plt.subplot(1, 2, 2)
selected_depths = [1, 3, None]
colors = ['blue', 'green', 'red']

for depth, color in zip(selected_depths, colors):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=random_seed)
    dt.fit(X_train, y_train)

    h = 0.02
    f1_min, f1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    f2_min, f2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, h), np.arange(f2_min, f2_max, h))

    Z = dt.predict(np.c_[f1.ravel(), f2.ravel()])
    Z = Z.reshape(f1.shape)

    depth_label = str(depth) if depth else 'None'
    plt.contour(f1, f2, Z, colors=color, alpha=0.5, linestyles='--',
                levels=[0.5], linewidths=2)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#0000FF', '#000000']), alpha=0.6)
plt.xlabel('特徵 1')
plt.ylabel('特徵 2')
plt.title('不同 max_depth 的決策邊界')
plt.legend(['depth=1', 'depth=3', 'depth=None'], loc='upper right')

plt.tight_layout()
plt.savefig(f'{output_dir}/decision_tree_depth_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ 圖片已儲存: decision_tree_depth_comparison.png")
plt.close()

# 4.3 在三個資料集上比較決策樹
print("\n[2.4] 在三個資料集上測試決策樹...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=random_seed)
plot_boundaries(dt_model, DataList, "決策樹 (Decision Tree, max_depth=5)",
                "decision_tree_boundaries.png")

# ============================
# 步驟5: 隨機森林 (Random Forest)
# ============================

print("\n" + "=" * 60)
print("[步驟3] 隨機森林 (Random Forest)")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# 5.1 n_estimators 比較
print("\n[3.1] n_estimators 參數比較...")

n_estimators_list = [5, 20, 100, 500]
rf_results = []

for n_est in n_estimators_list:
    print(f"\n   n_estimators = {n_est}:")

    for idx, (X, y) in enumerate(DataList):
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=5,
                                    random_state=random_seed)
        scores = cross_val_score(rf, X, y, cv=5)
        mean_score = scores.mean()

        rf_results.append({
            'n_estimators': n_est,
            '資料集': dataset_names[idx],
            '準確率': mean_score
        })
        print(f"      {dataset_names[idx]}: {mean_score:.4f}")

# 繪製 n_estimators 比較圖
plt.figure(figsize=(12, 5))

df_rf = pd.DataFrame(rf_results)

for idx, name in enumerate(dataset_names):
    plt.subplot(1, 3, idx+1)
    data = df_rf[df_rf['資料集'] == name]
    plt.plot(data['n_estimators'], data['準確率'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('n_estimators')
    plt.ylabel('準確率')
    plt.title(name)
    plt.ylim(0.7, 1.05)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

plt.suptitle('隨機森林 n_estimators 對準確率的影響', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/random_forest_n_estimators.png', dpi=150, bbox_inches='tight')
print("\n   ✓ 圖片已儲存: random_forest_n_estimators.png")
plt.close()

# 5.2 max_depth 比較
print("\n[3.2] max_depth 參數比較...")

max_depths_rf = [2, 5, 10, None]
rf_depth_results = []

for depth in max_depths_rf:
    depth_label = str(depth) if depth else 'None'
    print(f"\n   max_depth = {depth_label}:")

    for idx, (X, y) in enumerate(DataList):
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth,
                                    random_state=random_seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_seed
        )
        rf.fit(X_train, y_train)

        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)

        rf_depth_results.append({
            'max_depth': depth_label,
            '資料集': dataset_names[idx],
            '訓練準確率': train_acc,
            '測試準確率': test_acc
        })
        print(f"      {dataset_names[idx]}: 訓練={train_acc:.4f}, 測試={test_acc:.4f}")

# 繪製 max_depth 比較圖
plt.figure(figsize=(15, 5))

df_rf_depth = pd.DataFrame(rf_depth_results)

for idx, name in enumerate(dataset_names):
    plt.subplot(1, 3, idx+1)
    data = df_rf_depth[df_rf_depth['資料集'] == name]

    x_pos = range(len(max_depths_rf))
    width = 0.35

    plt.bar([p - width/2 for p in x_pos], data['訓練準確率'], width,
            label='訓練', color='skyblue')
    plt.bar([p + width/2 for p in x_pos], data['測試準確率'], width,
            label='測試', color='salmon')

    plt.xticks(x_pos, data['max_depth'])
    plt.xlabel('max_depth')
    plt.ylabel('準確率')
    plt.title(name)
    plt.legend()
    plt.ylim(0, 1.1)

plt.suptitle('隨機森林 max_depth 對準確率的影響', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/random_forest_max_depth.png', dpi=150, bbox_inches='tight')
print("\n   ✓ 圖片已儲存: random_forest_max_depth.png")
plt.close()

# 5.3 GridSearchCV 參數搜尋
print("\n[3.3] 使用 GridSearchCV 搜尋最佳參數...")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# 使用新月形資料集進行 GridSearchCV
X_moon, y_moon = DataList[1]

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=random_seed),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(X_moon, y_moon)

print(f"   最佳參數: {rf_grid.best_params_}")
print(f"   最佳分數: {rf_grid.best_score_:.4f}")

# 視覺化 GridSearchCV 結果
results_rf = pd.DataFrame(rf_grid.cv_results_)

plt.figure(figsize=(12, 5))

# 繪製 n_estimators vs max_depth 熱力圖
pivot_table = results_rf.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators',
    aggfunc='mean'
)

plt.subplot(1, 2, 1)
im = plt.imshow(pivot_table.values, cmap='YlGn', aspect='auto')
plt.colorbar(im, label='準確率')
plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.title('隨機森林 GridSearchCV 結果 (新月形資料集)')

# 在格子中顯示數值
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        plt.text(j, i, f'{pivot_table.values[i, j]:.3f}',
                ha='center', va='center', fontsize=9)

# 使用最佳參數繪製決策邊界
plt.subplot(1, 2, 2)
best_rf = rf_grid.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(
    X_moon, y_moon, test_size=0.3, random_state=random_seed
)

h = 0.02
f1_min, f1_max = X_moon[:, 0].min() - 0.5, X_moon[:, 0].max() + 0.5
f2_min, f2_max = X_moon[:, 1].min() - 0.5, X_moon[:, 1].max() + 0.5
f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, h), np.arange(f2_min, f2_max, h))

Z = best_rf.predict_proba(np.c_[f1.ravel(), f2.ravel()])[:, 1]
Z = Z.reshape(f1.shape)

plt.contourf(f1, f2, Z, cmap=plt.cm.bwr, alpha=0.3)
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap=ListedColormap(['#0000FF', '#000000']))
plt.xlabel('特徵 1')
plt.ylabel('特徵 2')
plt.title(f'最佳隨機森林 (準確率={rf_grid.best_score_:.4f})')

plt.tight_layout()
plt.savefig(f'{output_dir}/random_forest_gridsearch.png', dpi=150, bbox_inches='tight')
print("   ✓ 圖片已儲存: random_forest_gridsearch.png")
plt.close()

# 5.4 在三個資料集上測試隨機森林
print("\n[3.4] 在三個資料集上測試隨機森林...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_seed)
plot_boundaries(rf_model, DataList, "隨機森林 (Random Forest, n_estimators=100)",
                "random_forest_boundaries.png")

# ============================
# 步驟6: XGBoost
# ============================

print("\n" + "=" * 60)
print("[步驟4] XGBoost (Extreme Gradient Boosting)")
print("=" * 60)

try:
    from xgboost import XGBClassifier

    # 6.1 基本 XGBoost
    print("\n[4.1] 基本 XGBoost 模型...")

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_seed,
        eval_metric='logloss'
    )

    for idx, (X, y) in enumerate(DataList):
        scores = cross_val_score(xgb_model, X, y, cv=5)
        print(f"   {dataset_names[idx]}: {scores.mean():.4f} (±{scores.std():.4f})")

    # 6.2 n_estimators 參數調整
    print("\n[4.2] n_estimators 參數調整...")

    n_estimators_xgb = [50, 100, 200, 500]
    xgb_results = []

    for n_est in n_estimators_xgb:
        print(f"\n   n_estimators = {n_est}:")

        for idx, (X, y) in enumerate(DataList):
            xgb = XGBClassifier(n_estimators=n_est, max_depth=3,
                               learning_rate=0.1, random_state=random_seed,
                               eval_metric='logloss')
            scores = cross_val_score(xgb, X, y, cv=5)
            mean_score = scores.mean()

            xgb_results.append({
                'n_estimators': n_est,
                '資料集': dataset_names[idx],
                '準確率': mean_score
            })
            print(f"      {dataset_names[idx]}: {mean_score:.4f}")

    # 繪製 n_estimators 比較圖
    plt.figure(figsize=(12, 5))

    df_xgb = pd.DataFrame(xgb_results)

    for idx, name in enumerate(dataset_names):
        plt.subplot(1, 3, idx+1)
        data = df_xgb[df_xgb['資料集'] == name]
        plt.plot(data['n_estimators'], data['準確率'], 'o-', linewidth=2,
                markersize=8, color='green')
        plt.xlabel('n_estimators')
        plt.ylabel('準確率')
        plt.title(name)
        plt.ylim(0.7, 1.05)
        plt.grid(True, alpha=0.3)

    plt.suptitle('XGBoost n_estimators 對準確率的影響', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgboost_n_estimators.png', dpi=150, bbox_inches='tight')
    print("\n   ✓ 圖片已儲存: xgboost_n_estimators.png")
    plt.close()

    # 6.3 learning_rate 參數調整
    print("\n[4.3] learning_rate 參數調整...")

    learning_rates = [0.01, 0.1, 0.3, 0.5]
    xgb_lr_results = []

    X_moon, y_moon = DataList[1]

    for lr in learning_rates:
        xgb = XGBClassifier(n_estimators=100, max_depth=3,
                           learning_rate=lr, random_state=random_seed,
                           eval_metric='logloss')
        scores = cross_val_score(xgb, X_moon, y_moon, cv=5)
        mean_score = scores.mean()

        xgb_lr_results.append({
            'learning_rate': lr,
            '準確率': mean_score
        })
        print(f"   learning_rate={lr}: {mean_score:.4f}")

    # 6.4 GridSearchCV 參數搜尋
    print("\n[4.4] 使用 GridSearchCV 搜尋最佳參數...")

    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 5],
        'learning_rate': [0.01, 0.1, 0.3]
    }

    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=random_seed, eval_metric='logloss'),
        param_grid_xgb,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    xgb_grid.fit(X_moon, y_moon)

    print(f"   最佳參數: {xgb_grid.best_params_}")
    print(f"   最佳分數: {xgb_grid.best_score_:.4f}")

    # 視覺化 GridSearchCV 結果
    results_xgb = pd.DataFrame(xgb_grid.cv_results_)

    plt.figure(figsize=(15, 5))

    # 子圖1: n_estimators vs max_depth
    plt.subplot(1, 3, 1)
    pivot_1 = results_xgb.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators',
        aggfunc='mean'
    )
    im1 = plt.imshow(pivot_1.values, cmap='YlGn', aspect='auto')
    plt.colorbar(im1, label='準確率')
    plt.xticks(range(len(pivot_1.columns)), pivot_1.columns)
    plt.yticks(range(len(pivot_1.index)), pivot_1.index)
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.title('n_estimators vs max_depth')

    # 子圖2: learning_rate vs max_depth
    plt.subplot(1, 3, 2)
    pivot_2 = results_xgb.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_learning_rate',
        aggfunc='mean'
    )
    im2 = plt.imshow(pivot_2.values, cmap='YlGn', aspect='auto')
    plt.colorbar(im2, label='準確率')
    plt.xticks(range(len(pivot_2.columns)), pivot_2.columns)
    plt.yticks(range(len(pivot_2.index)), pivot_2.index)
    plt.xlabel('learning_rate')
    plt.ylabel('max_depth')
    plt.title('learning_rate vs max_depth')

    # 子圖3: 最佳模型決策邊界
    plt.subplot(1, 3, 3)
    best_xgb = xgb_grid.best_estimator_

    h = 0.02
    f1_min, f1_max = X_moon[:, 0].min() - 0.5, X_moon[:, 0].max() + 0.5
    f2_min, f2_max = X_moon[:, 1].min() - 0.5, X_moon[:, 1].max() + 0.5
    f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, h), np.arange(f2_min, f2_max, h))

    Z = best_xgb.predict_proba(np.c_[f1.ravel(), f2.ravel()])[:, 1]
    Z = Z.reshape(f1.shape)

    plt.contourf(f1, f2, Z, cmap=plt.cm.bwr, alpha=0.3)
    plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon,
               cmap=ListedColormap(['#0000FF', '#000000']))
    plt.xlabel('特徵 1')
    plt.ylabel('特徵 2')
    plt.title(f'最佳 XGBoost (準確率={xgb_grid.best_score_:.4f})')

    plt.suptitle('XGBoost GridSearchCV 結果', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgboost_gridsearch.png', dpi=150, bbox_inches='tight')
    print("   ✓ 圖片已儲存: xgboost_gridsearch.png")
    plt.close()

    # 6.5 在三個資料集上測試 XGBoost
    print("\n[4.5] 在三個資料集上測試 XGBoost...")
    xgb_final = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=random_seed, eval_metric='logloss')
    plot_boundaries(xgb_final, DataList, "XGBoost (n_estimators=100, max_depth=3)",
                    "xgboost_boundaries.png")

    xgboost_available = True

except ImportError:
    print("   ⚠ XGBoost 未安裝，跳過此演算法")
    print("   安裝指令: pip install xgboost")
    xgboost_available = False

# ============================
# 步驟7: 演算法比較總結
# ============================

print("\n" + "=" * 60)
print("[步驟5] 演算法比較總結")
print("=" * 60)

# 定義所有模型
models = {
    '決策樹 (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=random_seed),
    '決策樹 (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=random_seed),
    '決策樹 (depth=None)': DecisionTreeClassifier(max_depth=None, random_state=random_seed),
    '隨機森林 (n=50)': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_seed),
    '隨機森林 (n=100)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_seed),
    '隨機森林 (n=500)': RandomForestClassifier(n_estimators=500, max_depth=5, random_state=random_seed),
}

if xgboost_available:
    models['XGBoost (lr=0.01)'] = XGBClassifier(n_estimators=100, max_depth=3,
                                                 learning_rate=0.01, random_state=random_seed,
                                                 eval_metric='logloss')
    models['XGBoost (lr=0.1)'] = XGBClassifier(n_estimators=100, max_depth=3,
                                                learning_rate=0.1, random_state=random_seed,
                                                eval_metric='logloss')
    models['XGBoost (lr=0.3)'] = XGBClassifier(n_estimators=100, max_depth=3,
                                                learning_rate=0.3, random_state=random_seed,
                                                eval_metric='logloss')

# 比較所有模型
comparison_results = []

for dataset_idx, (X, y) in enumerate(DataList):
    print(f"\n資料集: {dataset_names[dataset_idx]}")
    print("-" * 50)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        mean_score = scores.mean()
        std_score = scores.std()

        comparison_results.append({
            '資料集': dataset_names[dataset_idx],
            '演算法': name,
            '平均準確率': mean_score,
            '標準差': std_score
        })

        print(f"   {name:20s}: {mean_score:.4f} (±{std_score:.4f})")

# 儲存結果
df_comparison = pd.DataFrame(comparison_results)
df_comparison.to_csv(f'{output_dir}/algorithm_comparison.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 比較結果已儲存至 algorithm_comparison.csv")

# 繪製比較圖
n_datasets = len(dataset_names)
n_models = len(models)

fig, axes = plt.subplots(1, n_datasets, figsize=(18, 8))

for idx, dataset_name in enumerate(dataset_names):
    ax = axes[idx]
    data = df_comparison[df_comparison['資料集'] == dataset_name]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(data)))

    bars = ax.barh(data['演算法'], data['平均準確率'], color=colors)
    ax.set_xlabel('準確率')
    ax.set_title(dataset_name, fontsize=14)
    ax.set_xlim(0, 1.05)

    # 加入數值標籤
    for bar, val in zip(bars, data['平均準確率']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)

plt.suptitle('樹狀分類演算法比較', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/algorithm_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 比較圖已儲存: algorithm_comparison.png")
plt.close()

# ============================
# 步驟8: 過擬合與泛化分析
# ============================

print("\n" + "=" * 60)
print("[步驟6] 過擬合與泛化分析")
print("=" * 60)

print("\n[6.1] 決策樹過擬合分析...")

# 使用不同 max_depth 分析過擬合
depths_analysis = list(range(1, 21))
train_scores = []
test_scores = []

X, y = DataList[1]  # 使用新月形資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

for depth in depths_analysis:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=random_seed)
    dt.fit(X_train, y_train)

    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

# 繪製過擬合分析圖
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(depths_analysis, train_scores, 'o-', label='訓練集', color='blue')
plt.plot(depths_analysis, test_scores, 's-', label='測試集', color='red')
plt.xlabel('max_depth')
plt.ylabel('準確率')
plt.title('決策樹: max_depth vs 準確率')
plt.legend()
plt.grid(True, alpha=0.3)

# 找出最佳 max_depth
best_depth = depths_analysis[np.argmax(test_scores)]
plt.axvline(x=best_depth, color='green', linestyle='--', label=f'最佳 depth={best_depth}')
plt.legend()

# 隨機森林 n_estimators 分析
print("\n[6.2] 隨機森林 n_estimators 分析...")

n_estimators_analysis = [1, 5, 10, 20, 50, 100, 200, 500]
rf_train_scores = []
rf_test_scores = []

for n_est in n_estimators_analysis:
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=5, random_state=random_seed)
    rf.fit(X_train, y_train)

    rf_train_scores.append(rf.score(X_train, y_train))
    rf_test_scores.append(rf.score(X_test, y_test))

plt.subplot(1, 2, 2)
plt.plot(n_estimators_analysis, rf_train_scores, 'o-', label='訓練集', color='blue')
plt.plot(n_estimators_analysis, rf_test_scores, 's-', label='測試集', color='red')
plt.xlabel('n_estimators')
plt.ylabel('準確率')
plt.title('隨機森林: n_estimators vs 準確率')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/overfitting_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ 圖片已儲存: overfitting_analysis.png")
plt.close()

# ============================
# 步驟9: 特徵重要性分析
# ============================

print("\n" + "=" * 60)
print("[步驟7] 特徵重要性分析 (使用 Iris 資料集)")
print("=" * 60)

# 訓練隨機森林並取得特徵重要性
rf_iris = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_seed)
rf_iris.fit(X_train_iris, y_train_iris)

feature_importance = rf_iris.feature_importances_
feature_names = iris.feature_names

# 排序特徵重要性
sorted_idx = np.argsort(feature_importance)[::-1]

print("\n特徵重要性排名:")
for i, idx in enumerate(sorted_idx):
    print(f"   {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

# 繪製特徵重要性圖
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx], color='skyblue')
plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx])
plt.xlabel('重要性')
plt.title('隨機森林特徵重要性 (Iris)')

# XGBoost 特徵重要性
if xgboost_available:
    xgb_iris = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=random_seed, eval_metric='logloss')
    xgb_iris.fit(X_train_iris, y_train_iris)

    xgb_importance = xgb_iris.feature_importances_
    xgb_sorted_idx = np.argsort(xgb_importance)[::-1]

    plt.subplot(1, 2, 2)
    plt.barh(range(len(xgb_importance)), xgb_importance[xgb_sorted_idx], color='lightgreen')
    plt.yticks(range(len(xgb_importance)), [feature_names[i] for i in xgb_sorted_idx])
    plt.xlabel('重要性')
    plt.title('XGBoost 特徵重要性 (Iris)')

plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n✓ 圖片已儲存: feature_importance.png")
plt.close()

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
print("Week 11 & 12 樹狀分類演算法練習完成！")
print("=" * 60)
