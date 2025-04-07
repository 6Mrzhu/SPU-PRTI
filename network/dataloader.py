import torch
from torch.utils.data import Dataset, DataLoader
import os


def parse_xyz_file(file_path):
    """
    解析.xyz文件，返回点的坐标列表。

    :param file_path: .xyz文件的路径
    :return: 点的坐标列表，每个点是一个(x, y, z)元组
    """
    with open(file_path, 'r') as file:
        points = []
        for line in file:
            x, y, z = line.strip().split()
            x, y, z = float(x), float(y), float(z)
            points.append((x, y, z))
    return points


class dataset(Dataset):
    def __init__(self, file_path):
        self.coordinates = parse_xyz_file(file_path)

    def __len__(self):
        # 这里假设整个文件是一个样本，返回1。如果文件中有多个样本，需要相应调整。
        return 1

    def __getitem__(self, idx):
        # 将坐标转换为 tensor
        coordinates_tensor = torch.tensor(self.coordinates, dtype=torch.float32)
        return coordinates_tensor


if __name__ =="__main__":
    file_name = 'poisson_256_00003.xyz'
    directory = '../network'
    file_path = os.path.join(directory, file_name)

  #  file_path = '.\ network\poisson_256_00003.xyz'
    dataset =dataset(file_path)
    dataset =dataset[0]
    print(dataset[0])