import pandas as pd
import time
from torch import Tensor, concat
from torch.utils.data import DataLoader, Dataset
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
    #print("cvt_dt input:\n", data)
    for feat in features:
        column = data.loc[:, feat[0]]
        #print(feat)
        data.loc[:, feat[0]] = column.map(convert_formats[feat[1]])
    #print("cvt_dt output:\n", data)
    return data

def open_dataset(directory: str, features: list[tuple[str, int]], output: str) -> tuple[Tensor, Tensor]:
    "return: inputs, outputs"
    data = pd.read_csv(directory).dropna()
    feature_str = [x[0] for x in features]
    select_data = data[feature_str]

    data_tensor = Tensor(convert_data(select_data, features).to_numpy(dtype=float32))
    out_index = feature_str.index(output)

    print("out index:", out_index, "tensor:", data_tensor.shape)
    inputs = concat((data_tensor[:, :out_index], data_tensor[:, (out_index + 1):]), 1)
    outputs = data_tensor[:, out_index, None]

    print("inputs:", inputs[0:3], inputs.shape)
    print("inputs:", outputs[0:3], outputs.shape)
    return inputs, outputs

class BookieDataset(Dataset):
    def __init__(self, inputs: Tensor, outputs: Tensor):
        self.inputs = inputs
        self.outputs = outputs
 
    def __len__(self):
        # this should return the size of the dataset
        return self.inputs.shape[0]
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.inputs[idx]
        target = self.outputs[idx]
        return features, target
    
def new_loader(inputs: Tensor, outputs: Tensor, batch_size: int) -> DataLoader:
    dataset = BookieDataset(inputs, outputs)
    return DataLoader(dataset, batch_size)
