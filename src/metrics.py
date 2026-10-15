"""
成员4：统一评测指标模块
功能：RMSE / MAE / MAPE / 方向准确率(DA) / 夏普比率
作者：成员4（评测与可视化负责人）
"""
import numpy as np


def calculate_rmse(y_true, y_pred):
    """均方根误差"""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_mae(y_true, y_pred):
    """平均绝对误差"""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_mape(y_true, y_pred):
    """平均绝对百分比误差（自动跳过真实值为0的样本）"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_da(y_true, y_pred):
    """
    方向准确率（Directional Accuracy）
    自动适配两种情况：
      1) 输入已经是收益率/差分 -> 直接比符号
      2) 输入是绝对价格 -> 先做一阶差分再比符号
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) < 2:
        return float('nan')

    # 判断是否需要差分：如果大部分值都为正（价格特征），则做差分
    if np.mean(y_true > 0) > 0.95:
        y_true = np.diff(y_true)
        y_pred = np.diff(y_pred)

    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)

    # 忽略真实方向为0（平盘）的样本
    valid = true_sign != 0
    if valid.sum() == 0:
        return float('nan')

    return float(np.mean(true_sign[valid] == pred_sign[valid]))


def calculate_sharpe(returns, risk_free=0.0, periods_per_year=252):
    """
    夏普比率（金融专属指标，用于评估交易价值）
    输入 returns 应为收益率序列
    """
    returns = np.asarray(returns).flatten()
    if len(returns) < 2 or np.std(returns) == 0:
        return float('nan')
    excess = returns - risk_free / periods_per_year
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def evaluate_all(y_true, y_pred, market="A", compute_sharpe=False):
    """
    统一评测入口，返回字典格式，供 Trainer / 论文图表调用。
    :param market: 'A' / 'US'，供你跨市场对比使用
    :param compute_sharpe: 是否计算夏普比率（需要预测的是收益率）
    """
    result = {
        "market": market,
        "RMSE": round(calculate_rmse(y_true, y_pred), 6),
        "MAE": round(calculate_mae(y_true, y_pred), 6),
        "MAPE": round(calculate_mape(y_true, y_pred), 4),
        "DA": round(calculate_da(y_true, y_pred), 4),
    }
    if compute_sharpe:
        result["Sharpe"] = round(calculate_sharpe(y_pred), 4)
    return result


if __name__ == "__main__":
    # 本地自测
    y_true = np.array([1.20, 1.30, 1.25, 1.40, 1.35, 1.50])
    y_pred = np.array([1.21, 1.28, 1.26, 1.38, 1.36, 1.48])
    print("绝对价格输入 ->", evaluate_all(y_true, y_pred, market="A"))
    print("收益率输入   ->", evaluate_all(np.diff(y_true), np.diff(y_pred), market="US"))