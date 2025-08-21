import qlib
from qlib.contrib.data.handler import Alpha158, Alpha360
from qlib.data import D
from qlib.constant import REG_CN
from qlib.data.dataset.processor import FilterCol, RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm
from qlib.data.dataset import TSDatasetH, DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP, DataHandler

start_time = '2005-01-04'
end_time = '2005-01-31'
fit_start_time = '2005-01-04'
fit_end_time = '2005-01-31'
market = ['SZ000001', 'SZ000002']

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
instruments = D.instruments(market=market)

data_handler_config = {
    "start_time": start_time,
    "end_time": end_time,
    "fit_start_time": fit_start_time,
    "fit_end_time": fit_end_time,
    "instruments": instruments,
    "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
}
handler = Alpha158(**data_handler_config)

print('----- raw -----')
df_raw = handler.fetch(col_set=handler.CS_RAW, data_key=DataHandler.DK_R)
print(df_raw['label'].head())

print('----- learn ----')
df_learning = handler.fetch(col_set=handler.CS_RAW, data_key=DataHandler.DK_L)
print(df_learning['label'].head())