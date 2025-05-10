import torch
from torch.utils.data import Dataset, DataLoader
import os


def parse_xyz_file(file_path):
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
       
        return 1

    def __getitem__(self, idx):
       
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
