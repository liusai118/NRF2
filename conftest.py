import pytest
from torch.utils.data import DataLoader
import myDataSet as ms

@pytest.fixture
def myDataLoader():
    # 示例数据
    input_data = ['ACGT', 'CGTA', 'GTAC']  # 这里替换为实际的输入数据
    label_data = [0, 1, 0]  # 这里替换为实际的标签数据
    dataset = ms.MyDataSet(input=input_data, label=label_data)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader
