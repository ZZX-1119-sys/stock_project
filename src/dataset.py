import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """
    股票时序数据集，支持灵活设置回看窗口和预测步长。
    
    参数:
        data: DataFrame，包含 date, code, 以及特征列
        lookback_len: 回看窗口长度（如 96/192/336/720）
        pred_len: 预测步长（如 1 表示预测下一天）
        feature_cols: 特征列列表，None 表示自动排除非特征列
        target_col: 预测目标列，默认 'close'
    """
    
    def __init__(self, data, lookback_len=96, pred_len=1,
                 feature_cols=None, target_col='close'):
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.target_col = target_col
        
        # 默认特征列：只排除 date 和 code，保留 close 作为输入特征
        if feature_cols is None:
            exclude = ['date', 'code']
            self.feature_cols = [c for c in data.columns if c not in exclude]
        else:
            self.feature_cols = feature_cols
        
        # 按股票分组，构造样本索引
        self.samples = []
        for code, group in data.groupby('code'):
            group = group.sort_values('date').reset_index(drop=True)
            n = len(group)
            for i in range(n - lookback_len - pred_len + 1):
                self.samples.append((code, i))
        
        # 存储数据，避免每个样本重复切片
        self.data = data.sort_values(['code', 'date']).reset_index(drop=True)
        self.group_indices = {}
        for code, group in self.data.groupby('code'):
            self.group_indices[code] = group.index.tolist()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        code, i = self.samples[idx]
        indices = self.group_indices[code]
        
        # 回看窗口
        start = indices[i]
        end = indices[i + self.lookback_len]
        features = self.data.loc[start:end-1, self.feature_cols].values
        
        # 预测目标
        target_start = end
        target_end = end + self.pred_len
        targets = self.data.loc[target_start:target_end-1, self.target_col].values
        
        features = torch.FloatTensor(features.astype(np.float32))
        targets = torch.FloatTensor(targets.astype(np.float32))
        
        return features, targets