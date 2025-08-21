import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from qlib.log import get_module_logger
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

class GRUAttentionDataSampler(Dataset):
    """
    一个用于处理股票数据的PyTorch Dataset类。
    
    该类将原始DataFrame数据按时间窗口（step_len）划分，并将所有股票在每个时间步的数据组织成一个三维张量。

    Args:
        df (pd.DataFrame): 包含datetime, stock_id, label和特征列的DataFrame。
        step_len (int): 每个样本的时间窗口大小，即输入GRU的days。
        feature_cols (list): 包含所有特征列名的列表。
        label_col (str): 标签列的名称。
    """
    def __init__(self, df: pd.DataFrame, step_len: int = 30, regression: bool = True):
        self.logger = get_module_logger("GRUAttentionDataSampler")
        self.step_len = step_len
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.samples = []
        has_label = True if 'label' in df.columns else False

        # 确保数据已按时间排序
        df = df.sort_index()
        
        # 获取唯一的日期和股票ID
        unique_dates = df.index.get_level_values("datetime").unique()
        unique_stocks = df.index.get_level_values('instrument').unique()
        self.logger.info(f"开始处理数据，共有 {len(unique_dates)} 个时间步，{len(unique_stocks)} 个股票。")
        
        # 建立一个日期到索引的映射
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        
        # 创建一个三维的特征数组 (days, stocks, features)
        all_features = np.zeros(
            (len(unique_dates), len(unique_stocks), df['feature'].shape[1]), 
            dtype=np.float32
        )
        # 记录该 (date, stock) 是否在原始数据中出现（用于动态股票池/未上市）
        presence = np.zeros((len(unique_dates), len(unique_stocks)), dtype=bool)
        
        if has_label:
            label_type = np.float32 if regression else np.int64
            all_labels = np.zeros((len(unique_dates), len(unique_stocks)), dtype=label_type)

        # 填充三维数组
        for stock_id_idx, stock_id in enumerate(tqdm(unique_stocks, desc="填充数据数组")):
            stock_df = df.xs(stock_id, level='instrument')
            for _, row in stock_df.iterrows():
                date_idx = date_to_idx[row.name]
                all_features[date_idx, stock_id_idx, :] = row['feature'].values
                presence[date_idx, stock_id_idx] = True
                if has_label:
                    all_labels[date_idx, stock_id_idx] = row['label'].values[0]

        # 基于原始数值构造 feature_mask：要求该位置存在且所有特征为有限值（保持为 bool 以节省内存）
        finite_feat = np.isfinite(all_features).all(axis=-1)
        feature_mask_full = (presence & finite_feat)  # (days, stocks) -> bool
        # label 的有效性：存在且为有限值（分类任务的整型也视为有效），保持为 bool
        if has_label:
            label_finite = np.isfinite(all_labels)
            label_mask_full = (presence & label_finite)  # bool
        else:
            label_mask_full = None

        # 统计原始 all_features 中 NaN/Inf 的占比（用于诊断）
        nan_ratio = np.isnan(all_features).mean()
        inf_ratio = np.isinf(all_features).mean()
        pres_ratio = presence.mean()
        self.logger.info(
            f"features: NaN {nan_ratio:.4f}, Inf {inf_ratio:.4f}, presence {pres_ratio:.4f}, feature_mask mean {feature_mask_full.mean():.4f}"
        )

        # 遍历所有时间步，创建时间窗口样本
        for i in tqdm(range(len(unique_dates) - self.step_len + 1), desc="生成样本"):
            # 获取当前窗口的特征和标签
            window_features = all_features[i:i + self.step_len, :, :]
            window_feature_mask = feature_mask_full[i:i + self.step_len, :]
            features_tensor = torch.tensor(window_features, dtype=torch.float32)
            # 保持 mask 为 bool，避免额外内存占用
            feature_mask_tensor = torch.from_numpy(window_feature_mask.astype(np.bool_))

            if has_label:
                label_type = torch.float32 if regression else torch.long
                window_label = all_labels[i + self.step_len - 1, :]
                label_tensor = torch.tensor(window_label, dtype=label_type)
                # 仅最后一天用于监督的 label_mask
                window_label_mask = label_mask_full[i + self.step_len - 1, :]
                label_mask_tensor = torch.from_numpy(window_label_mask.astype(np.bool_))
                # 返回：(features, label, feature_mask, label_mask)
                self.samples.append((features_tensor, label_tensor, feature_mask_tensor, label_mask_tensor))
            else:
                # 返回：(features, feature_mask)
                self.samples.append((features_tensor, feature_mask_tensor))
            
        self.logger.info(f"数据处理完成，共生成 {len(self.samples)} 个时间步的样本。device: {self.samples[0][0].device if self.samples else 'N/A'}")

    def __len__(self):
        """
        返回数据集中样本的总数（即时间步数）。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本。
        
        返回:
            (window_features, window_label): 
            window_features 形状为 (days, stocks, features)
            window_label 形状为 (stocks,)
        """
        return self.samples[idx]
    
# Qlib-style Dataset for GRU with Attention Model
class GRUAttentionDatasetH(DatasetH):
    """
    Dataset for GRU with Cross-stock Attention Model
    
    This dataset extends DatasetH to provide cross-sectional data structure 
    needed for attention mechanism across stocks at each time step.
    
    The dataset reshapes time-series data into format:
    (batch_size, days, stocks, features) for attention computation.
    """
    
    def __init__(self, step_len: int = 30, regression: bool = True, **kwargs):
        """
        Parameters
        ----------
        step_len : int
            Length of the time series lookback window
        num_stocks : int, optional
            Expected number of stocks. If None, will be inferred from data
        """
        self.logger = get_module_logger("GRUAttentionDatasetH")
        self.step_len = step_len
        self.regression = regression
        super().__init__(**kwargs)
    
    def _prepare_seg(self, slc, **kwargs) -> GRUAttentionDataSampler:
        """
        Prepare data segment for GRU attention model
        
        Returns
        -------
        GRUAttentionDataSampler
            Custom data sampler that provides (days, stocks, features) shaped data
        """
        if not isinstance(slc, slice):
            slc = slice(*slc)
        
        # data = self.handler.fetch(col_set=["feature", "label"], data_key=kwargs["data_key"])
        df = super()._prepare_seg(slc, **kwargs)
        # self.logger.info(f"--------- Data prepared for segment {slc} with shape: {df.shape}, head: {df.head()}, tail: {df.tail()}, kwargs: {kwargs}")

        df_feature = df["feature"]
        nan_ratio = df_feature.isna().mean().mean()
        
        # 计算总元素数量
        total_elements = df_feature.size
        # 检测无穷大值
        inf_mask = np.isinf(df_feature)
        # 计算无穷大值的数量
        inf_elements = inf_mask.sum().sum()
        # 计算无穷大值所占比例
        inf_ratio = inf_elements / total_elements
        nan_cols = df_feature.columns[(df_feature.isna().any())]
        self.logger.info(f"NaN ratio in df_feature {nan_ratio:.4f}, inf ratio: {inf_ratio:.4f}, cols has nan: {nan_cols}")

        # # 按 datetime 分组并计算每组的大小
        # group_sizes = df.groupby(level='datetime').size()
        # # 检查所有组的大小是否相同
        # all_same = group_sizes.nunique() == 1
        # print("\n每个 datetime 的组大小:")
        # print(group_sizes)
        # print("\n所有 datetime 的组大小是否相同:", all_same)

        # 如果是推理，则记录df的index，用于在推理时恢复原始数据，记录时只需从datetime index的step_len-1开始
        if kwargs["data_key"] == DataHandlerLP.DK_I:
            unique_datetimes = df.index.get_level_values('datetime').unique()
            start_datetime = unique_datetimes[self.step_len - 1]
            self.infer_index = df.index[df.index.get_level_values('datetime') >= start_datetime]
            self.logger.info(f"--------- infer_index head: {self.infer_index}")

        return GRUAttentionDataSampler(df=df, step_len=self.step_len, regression=self.regression)

# class GRUAttentionDataSampler2:
#     """
#     Data sampler for GRU Attention model
    
#     Provides data in shape (batch_size, days, stocks, features) format
#     required by GRUWithAttentionModel.
#     """
    
#     def __init__(self, 
#                  data: pd.DataFrame,
#                  start, 
#                  end,
#                  step_len: int,
#                  num_stocks: int,
#                  instruments: List[str]):
#         """
#         Parameters
#         ----------
#         data : pd.DataFrame
#             Raw data with MultiIndex (datetime, instrument)
#         start, end : 
#             Start and end time for the target period
#         step_len : int
#             Length of time series lookback
#         num_stocks : int
#             Number of stocks expected
#         instruments : List[str]
#             List of instrument codes
#         """
#         self.logger = get_module_logger("GRUAttentionDataSampler")
#         self.start = start
#         self.end = end  
#         self.step_len = step_len
#         self.num_stocks = num_stocks
#         self.instruments = instruments
        
#         # Keep original (datetime, instrument) indexing for proper date-based access
#         self.data = data.sort_index()
        
#         # Get unique dates in the data
#         self.dates = sorted(self.data.index.get_level_values('datetime').unique())
#         self.logger.info(f"start date: {self.dates[0]}, end date: {self.dates[-1]}")

#         # Filter dates to target period
#         start_date = pd.Timestamp(start)
#         end_date = pd.Timestamp(end)
#         self.target_dates = [d for d in self.dates if start_date <= d <= end_date]
        
#         # Build cross-sectional samples
#         self._build_samples()
    
#     def _build_samples(self):
#         """Build samples in (days, stocks, features) format"""
#         self.samples = []
#         self.sample_indices = []
        
#         # Get feature columns (assuming 'feature' is the main column group)
#         if 'feature' in self.data.columns:
#             feature_data = self.data['feature']
#         else:
#             feature_data = self.data
            
#         # Get label columns if available
#         if 'label' in self.data.columns:
#             label_data = self.data['label']
#         else:
#             label_data = None
        
#         for i, target_date in enumerate(self.target_dates):
#             # Important: target_dates are filtered to be within [start, end] range
#             # This ensures we only create samples for the intended time period
#             # But we can use historical data from the extended dataset for lookback
            
#             # Find the index of target date in all available dates
#             start_idx = self.dates.index(target_date)
#             end_idx = start_idx + self.step_len
#             if end_idx > len(self.dates):
#                 break
#             hist_dates = self.dates[start_idx:end_idx]

#             # Build cross-sectional data: (days, stocks, features)
#             sample_features = []
            
#             for date in hist_dates:
#                 # Get all stocks data for this date using proper datetime-based indexing
#                 try:
#                     date_data = feature_data.loc[date]
#                     if isinstance(date_data, pd.Series):
#                         # Only one stock for this date
#                         date_features = date_data.values.reshape(1, -1)
#                     else:
#                         # Multiple stocks
#                         date_features = date_data.values
                    
#                     # Ensure we have data for all expected stocks
#                     if len(date_features) < self.num_stocks:
#                         raise ValueError("Insufficient stocks for feature")
#                     elif len(date_features) > self.num_stocks:
#                         raise ValueError("Too much stocks for feature")
                        
#                 except KeyError:
#                     raise ValueError(f"No data available for date: {date}")
                
#                 sample_features.append(date_features)
            
#             if len(sample_features) < self.step_len:
#                 raise ValueError("Insufficient historical data for the specified step_len")
            
#             sample_feature_array = np.stack(sample_features)  # (days, stocks, features)
            
#             # Get labels for the target date if available
#             if label_data is not None:
#                 try:
#                     target_labels = label_data.loc[hist_dates[-1]].values
#                     if target_labels.ndim == 0:
#                         target_labels = target_labels.reshape(1)
#                     if len(target_labels) < self.num_stocks:
#                         raise ValueError("Insufficient stocks for label")
#                     elif len(target_labels) > self.num_stocks:
#                         raise ValueError("Too much stocks for label")
#                 except KeyError:
#                     raise ValueError(f"No data available for date: {hist_dates[-1]}")
#             else:
#                 target_labels = np.full(self.num_stocks, np.nan)
            
#             self.samples.append((sample_feature_array, target_labels))
            
#             # Create MultiIndex for this sample (for compatibility with qlib)
#             sample_index = [(target_date, inst) for inst in self.instruments]
#             self.sample_indices.append(sample_index)
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         if isinstance(idx, int):
#             return self.samples[idx]
#         elif isinstance(idx, slice):
#             return [self.samples[i] for i in range(*idx.indices(len(self.samples)))]
#         else:
#             raise TypeError(f"Invalid index type: {type(idx)}")
    
#     def get_index(self):
#         """Get MultiIndex for all samples (compatibility with qlib)"""
#         all_indices = []
#         for sample_idx in self.sample_indices:
#             all_indices.extend(sample_idx)
#         return pd.MultiIndex.from_tuples(all_indices, names=['datetime', 'instrument'])
    
#     @property
#     def empty(self):
#         return len(self.samples) == 0