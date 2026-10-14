"""
成员4：论文级可视化工具包
功能：预测对比图 / 误差分布图 / 特征重要性图 / 跨市场对比图
作者：成员4（评测与可视化负责人）
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 全局风格（论文级） ----------
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style="ticks", palette="colorblind")

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _save(save_path):
    """统一保存：自动建目录 + 高分辨率"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plotter] 图已保存: {save_path}")


def plot_prediction_vs_groundtruth(y_true, y_pred,
                                   save_path=f"{FIG_DIR}/pred_vs_true.png",
                                   title="Prediction vs Ground Truth"):
    """预测值 vs 真实值 折线图"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Ground Truth', color='black', linewidth=1.5)
    plt.plot(y_pred, label='Prediction', color='#D55E00', linestyle='--', linewidth=1.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    sns.despine()
    _save(save_path)


def plot_error_distribution(y_true, y_pred,
                            save_path=f"{FIG_DIR}/error_distribution.png"):
    """误差分布直方图 + 箱线图"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(errors, bins=50, kde=True, ax=axes[0], color='#0072B2')
    axes[0].set_title('Error Distribution')
    axes[0].set_xlabel('Prediction Error')

    sns.boxplot(x=errors, ax=axes[1], color='#E69F00')
    axes[1].set_title('Error Boxplot')
    axes[1].set_xlabel('Prediction Error')

    sns.despine()
    _save(save_path)


def plot_feature_importance(feature_names, importance_scores,
                            save_path=f"{FIG_DIR}/feature_importance.png",
                            title="Feature Importance"):
    """
    单市场特征重要性柱状图
    用于论文4：置换重要性（Permutation Importance）可视化
    """
    feature_names = list(feature_names)
    importance_scores = np.asarray(importance_scores).flatten()

    sorted_idx = np.argsort(importance_scores)
    sorted_names = np.array(feature_names)[sorted_idx]
    sorted_scores = importance_scores[sorted_idx]

    plt.figure(figsize=(8, max(4, 0.4 * len(feature_names))))
    plt.barh(sorted_names, sorted_scores, color='#0072B2')
    plt.xlabel('Permutation Importance (MSE Increase)')
    plt.title(title)
    sns.despine()
    _save(save_path)


def plot_cross_market_importance(feature_names, a_importance, us_importance,
                                  save_path=f"{FIG_DIR}/cross_market_importance.png",
                                  title="Feature Importance: A-Share vs US-Share"):
    """
    跨市场双柱状对比图 —— 论文4核心图表
    :param feature_names: 特征名列表（两市场须一致）
    :param a_importance: A股特征重要性
    :param us_importance: 美股特征重要性
    """
    feature_names = list(feature_names)
    a_importance = np.asarray(a_importance).flatten()
    us_importance = np.asarray(us_importance).flatten()

    assert len(feature_names) == len(a_importance) == len(us_importance), \
        "feature_names / a_importance / us_importance 长度必须一致"

    x = np.arange(len(feature_names))
    width = 0.38

    plt.figure(figsize=(10, max(5, 0.45 * len(feature_names))))
    plt.bar(x - width / 2, a_importance, width, label='A-Share', color='#E69F00')
    plt.bar(x + width / 2, us_importance, width, label='US-Share', color='#56B4E9')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.ylabel('Permutation Importance')
    plt.title(title)
    plt.legend()
    sns.despine()
    _save(save_path)


if __name__ == "__main__":
    # 本地自测：生成 4 张图
    y_true = np.array([1.20, 1.30, 1.25, 1.40, 1.35, 1.50, 1.45, 1.55])
    y_pred = np.array([1.21, 1.28, 1.26, 1.38, 1.36, 1.48, 1.47, 1.53])

    plot_prediction_vs_groundtruth(y_true, y_pred)
    plot_error_distribution(y_true, y_pred)

    features = ['open', 'high', 'low', 'close', 'volume',
                'macd', 'rsi', 'boll_upper', 'cpi', 'm2', 'lpr']
    a_imp = [0.45, 0.30, 0.28, 0.50, 0.12, 0.08, 0.05, 0.03, 0.06, 0.04, 0.02]
    us_imp = [0.40, 0.28, 0.25, 0.48, 0.15, 0.10, 0.07, 0.04, 0.09, 0.11, 0.13]
    plot_feature_importance(features, a_imp)
    plot_cross_market_importance(features, a_imp, us_imp)