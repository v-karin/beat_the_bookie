import pandas as pd
import time
from torch import Tensor, concat
from torch.utils.data import DataLoader
from numpy import float32

FORMAT_TYPES = ["int", "name", "date", "match"]
INT = 0
NAME = 1
DATE = 2
MATCH = 3

def convert_int(x): return x
convert_name = hash
def convert_date(date):
    if len(date) == 8:
        return time.mktime(time.strptime(date, "%d/%m/%y")) # 2 letter year
    elif len(date) == 10:
        return time.mktime(time.strptime(date, "%d/%m/%Y")) # 4 letter year
    else:
        raise Exception("Date:", date, "incompatible format.")
convert_match = {"H": -1, "D": 0, "A": 1}

convert_formats = [
    convert_int,
    convert_name,
    convert_date,
    convert_match
]

def convert_data(data: pd.DataFrame, features: list[tuple[str, int]]) -> pd.DataFrame:
    print("cvt_dt input:\n", data)
    for feat in features:
        column = data.loc[:, feat[0]]
        print(feat)
        data.loc[:, feat[0]] = column.map(convert_formats[feat[1]])
    print("cvt_dt output:\n", data)
    return data

def open_dataset(directory: str, features: list[tuple[str, int]], output: str) -> tuple[Tensor, Tensor]:
    "returns: inputs, outputs"
    data = pd.read_csv(directory).dropna()
    feature_str = [x[0] for x in features]
    select_data = data[feature_str]
    data_tensor = Tensor(convert_data(select_data, features).to_numpy(dtype=float32))
    out_index = feature_str.index(output)
    print("out index:", out_index, "tensor:", data_tensor.shape)
    inputs = concat((data_tensor[:, :out_index], data_tensor[:, (out_index + 1):]), 1)
    outputs = data_tensor[:, out_index]
    print("inputs, outputs:", inputs.shape, outputs.shape)
    return inputs, outputs

