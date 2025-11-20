"""
大數據資料分析實作 - Week 10 分類模型演算法進階練習
Classification Model Algorithms Advanced Practice

本腳本實作 Week 10 PDF 中的所有進階練習：
1. 激活函數視覺化 (Sigmoid, Tanh, ReLU)
2. SVM 核函數比較 (Linear, Poly, RBF, Sigmoid)
3. SVM C 參數調整
4. SVM Gamma 參數調整
5. GridSearchCV 最佳參數搜尋
6. 神經網路架構比較
"""

# ============================
# 步驟1: 安裝與設定
# ============================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 設定隨機種子
random_seed = 123

np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)

# 中文字體設定 (跨平台)
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

print("=" * 70)
print("大數據資料分析實作 - Week 10 分類模型演算法進階練習")
print("=" * 70)

# 確保 outputs 資料夾存在
import os
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
    print("\n✓ 已創建 outputs 資料夾")

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
# 步驟3: 定義輔助函式
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

# ============================
# 練習 1: 激活函數視覺化
# ============================

print("\n" + "=" * 70)
print("[練習 1] 激活函數視覺化")
print("=" * 70)

# 定義激活函數
def sigmoid(x):
    """Sigmoid 函數"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh 函數 (雙曲正切)"""
    return np.tanh(x)

def relu(x):
    """ReLU 函數 (Rectified Linear Unit)"""
    return np.maximum(0, x)

# 生成 x 值
x = np.linspace(-5, 5, 100)

# 計算各激活函數的值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

print("\n繪製激活函數比較圖...")

# 繪製三個激活函數的比較圖
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sigmoid
axes[0].plot(x, y_sigmoid, 'b-', linewidth=2)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Sigmoid 函數', fontsize=16)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('sigmoid(x)', fontsize=12)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].text(0.5, 0.5, 'σ(x) = 1/(1+e⁻ˣ)', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Tanh
axes[1].plot(x, y_tanh, 'r-', linewidth=2)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Tanh 函數', fontsize=16)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('tanh(x)', fontsize=12)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1].text(0.5, 0.7, 'tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ReLU
axes[2].plot(x, y_relu, 'g-', linewidth=2)
axes[2].grid(True, alpha=0.3)
axes[2].set_title('ReLU 函數', fontsize=16)
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('ReLU(x)', fontsize=12)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[2].text(1, 3, 'ReLU(x) = max(0, x)', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./outputs/activation_functions_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 激活函數比較圖已儲存: activation_functions_comparison.png")
plt.close()

# ============================
# 練習 2: SVM 核函數比較
# ============================

print("\n" + "=" * 70)
print("[練習 2] SVM 核函數比較")
print("=" * 70)

from sklearn.svm import SVC

# 定義四種核函數
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_names = {
    'linear': '線性核 (Linear)',
    'poly': '多項式核 (Polynomial)',
    'rbf': '高斯核 (RBF)',
    'sigmoid': 'Sigmoid核'
}

dataset_names = ['線性可分離', '新月形', '同心圓']

print("\n測試 4 種核函數 × 3 種資料集 = 12 種組合...")

# 創建大圖表
fig, axes = plt.subplots(4, 3, figsize=(18, 22))

for i, kernel in enumerate(kernels):
    print(f"\n{kernel_names[kernel]}:")
    for j, (X, y) in enumerate(DataList):
        ax = axes[i, j]

        # 創建 SVM 模型
        if kernel == 'poly':
            model = SVC(kernel=kernel, degree=3, random_state=random_seed)
        else:
            model = SVC(kernel=kernel, gamma='auto', random_state=random_seed)

        # 繪製決策邊界
        plot_boundary(ax, X, y, model)

        # 設定標題
        if i == 0:
            ax.set_title(dataset_names[j], fontsize=14, fontweight='bold')

        if j == 0:
            ax.set_ylabel(kernel_names[kernel], fontsize=14, fontweight='bold')

        print(f"  ✓ {dataset_names[j]}")

plt.tight_layout()
plt.savefig('./outputs/svm_kernels_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ SVM 核函數比較圖已儲存: svm_kernels_comparison.png")
plt.close()

# ============================
# 練習 3: SVM C 參數調整
# ============================

print("\n" + "=" * 70)
print("[練習 3] SVM C 參數調整")
print("=" * 70)

# C 參數：懲罰參數，控制錯誤分類的容忍度
C_values = [0.01, 0.1, 1, 10, 100]

print("\n測試不同 C 值對決策邊界的影響...")
print("使用資料集: 新月形 (Moon)")
print("使用核函數: RBF")

# 使用新月形資料集
X, y = X2, y2

fig, axes = plt.subplots(1, 5, figsize=(25, 4))

for idx, C in enumerate(C_values):
    ax = axes[idx]

    # 創建 SVM 模型
    model = SVC(kernel='rbf', C=C, gamma='auto', random_state=random_seed)

    # 繪製決策邊界
    plot_boundary(ax, X, y, model)

    ax.set_title(f'C = {C}', fontsize=14, fontweight='bold')

    print(f"✓ C = {C}")

plt.suptitle('SVM C 參數調整 (C 越大，對錯誤分類的懲罰越大)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('./outputs/svm_c_parameter_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ SVM C 參數比較圖已儲存: svm_c_parameter_comparison.png")
plt.close()

# ============================
# 練習 4: SVM Gamma 參數調整
# ============================

print("\n" + "=" * 70)
print("[練習 4] SVM Gamma 參數調整")
print("=" * 70)

# Gamma 參數：核函數係數，控制單一訓練樣本的影響範圍
gamma_values = [0.01, 0.1, 1, 10, 100]

print("\n測試不同 Gamma 值對決策邊界的影響...")
print("使用資料集: 新月形 (Moon)")
print("使用核函數: RBF")

fig, axes = plt.subplots(1, 5, figsize=(25, 4))

for idx, gamma in enumerate(gamma_values):
    ax = axes[idx]

    # 創建 SVM 模型
    model = SVC(kernel='rbf', C=1, gamma=gamma, random_state=random_seed)

    # 繪製決策邊界
    plot_boundary(ax, X, y, model)

    ax.set_title(f'Gamma = {gamma}', fontsize=14, fontweight='bold')

    print(f"✓ Gamma = {gamma}")

plt.suptitle('SVM Gamma 參數調整 (Gamma 越大，單一樣本影響範圍越小)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('./outputs/svm_gamma_parameter_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ SVM Gamma 參數比較圖已儲存: svm_gamma_parameter_comparison.png")
plt.close()

# ============================
# 練習 5: GridSearchCV 最佳參數搜尋
# ============================

print("\n" + "=" * 70)
print("[練習 5] GridSearchCV 最佳參數搜尋")
print("=" * 70)

from sklearn.model_selection import GridSearchCV

print("\n使用 GridSearchCV 搜尋最佳 C 和 Gamma 組合...")
print("搜尋範圍:")
print(f"  C: {C_values}")
print(f"  Gamma: {gamma_values}")
print("交叉驗證: 5-fold")

# 定義參數網格
param_grid = {
    'C': C_values,
    'gamma': gamma_values
}

# 創建 GridSearchCV
grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=random_seed),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# 在新月形資料集上進行搜尋
grid_search.fit(X2, y2)

print("\n✓ GridSearchCV 完成！")
print(f"\n最佳參數:")
print(f"  C = {grid_search.best_params_['C']}")
print(f"  Gamma = {grid_search.best_params_['gamma']}")
print(f"\n最佳準確率: {grid_search.best_score_:.4f}")

# 視覺化最佳模型
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左圖：預設參數
ax1 = axes[0]
model_default = SVC(kernel='rbf', gamma='auto', random_state=random_seed)
plot_boundary(ax1, X2, y2, model_default)
ax1.set_title('預設參數 (C=1, gamma=auto)', fontsize=14, fontweight='bold')

# 右圖：最佳參數
ax2 = axes[1]
model_best = grid_search.best_estimator_
plot_boundary(ax2, X2, y2, model_best)
ax2.set_title(f'最佳參數 (C={grid_search.best_params_["C"]}, '
              f'gamma={grid_search.best_params_["gamma"]})',
              fontsize=14, fontweight='bold')

plt.suptitle('GridSearchCV - 最佳參數 vs 預設參數', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('./outputs/svm_gridsearch_best_model.png', dpi=150, bbox_inches='tight')
print("✓ GridSearchCV 結果圖已儲存: svm_gridsearch_best_model.png")
plt.close()

# 將所有參數組合的結果儲存為 CSV
results_df = pd.DataFrame(grid_search.cv_results_)
results_summary = results_df[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score']]
results_summary.columns = ['C', 'Gamma', '平均準確率', '標準差']
results_summary = results_summary.sort_values('平均準確率', ascending=False)
results_summary.to_csv('./outputs/gridsearch_results.csv', index=False, encoding='utf-8-sig')
print("✓ GridSearchCV 詳細結果已儲存: gridsearch_results.csv")

# ============================
# 練習 6: 神經網路架構比較
# ============================

print("\n" + "=" * 70)
print("[練習 6] 神經網路架構比較")
print("=" * 70)

from sklearn.neural_network import MLPClassifier

print("\n比較不同神經網路架構...")

# 定義兩種架構
architectures = {
    '預設架構 (1層, 100神經元)': MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=random_seed,
        alpha=0.01
    ),
    '自訂架構 (2層, 100+100神經元)': MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=1000,
        random_state=random_seed,
        alpha=0.01
    )
}

# 創建圖表
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

for i, (arch_name, model) in enumerate(architectures.items()):
    print(f"\n{arch_name}:")
    for j, (X, y) in enumerate(DataList):
        ax = axes[i, j]

        # 繪製決策邊界
        plot_boundary(ax, X, y, model)

        # 設定標題
        if i == 0:
            ax.set_title(dataset_names[j], fontsize=14, fontweight='bold')

        if j == 0:
            ax.set_ylabel(arch_name, fontsize=12, fontweight='bold')

        print(f"  ✓ {dataset_names[j]}")

plt.suptitle('神經網路架構比較', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig('./outputs/mlp_architecture_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ 神經網路架構比較圖已儲存: mlp_architecture_comparison.png")
plt.close()

# ============================
# 總結報告
# ============================

print("\n" + "=" * 70)
print("所有練習完成！")
print("=" * 70)

print("\n生成的檔案:")
print("  1. activation_functions_comparison.png    - 激活函數比較")
print("  2. svm_kernels_comparison.png             - SVM 核函數比較")
print("  3. svm_c_parameter_comparison.png         - SVM C 參數調整")
print("  4. svm_gamma_parameter_comparison.png     - SVM Gamma 參數調整")
print("  5. svm_gridsearch_best_model.png          - GridSearchCV 最佳模型")
print("  6. gridsearch_results.csv                 - GridSearchCV 詳細結果")
print("  7. mlp_architecture_comparison.png        - 神經網路架構比較")

print("\n✓ Week 10 所有練習完成！")
print("=" * 70)
