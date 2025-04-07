import torch.nn as nn

import torch
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import os
#from utils.pc_util import downsample_points,point_cloud_to_volume_batch,pyplot_draw_point_cloud
from sklearn.neighbors import KDTree

import open3d as o3d
import numpy as np
# from torch.utils.data import Dataset, DataLoader
from chamfer_dist import ChamferFunction as chamfer_3DDist
chamfer_distance1 = chamfer_3DDist.apply
from dataloader import dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
def downsample_points(pts, K):
    # if num_pts > 8K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2*K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K,
            replace=(K<pts.shape[0])), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts

def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
   # distance = torch.ones(B, N).to(device) * 1e10
    distance = torch.full((B, N), 1e10, dtype=torch.float32, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return  centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
def read_point_cloud(file_path):
    """
    读取点云数据，假设每行格式为 'x y z'。

    参数:
    file_path (str): 点云数据文件的路径。

    返回:
    points (list of tuples): 包含点云数据的列表，每个元素是一个 (x, y, z) 元组。
    """
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行末的换行符和可能的空白字符
            line = line.strip()
            # 跳过空行
            if not line:
                continue
                # 分割行，并转换为浮点数
            parts = line.split()
            if len(parts) == 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                points.append((x, y, z))
            else:
                print(f"警告: 跳过格式不正确的行: {line}")
    return points

#判断离散点并进行删除
def remove_outliers(xyzout,k,a):
     device = xyzout.device
     xyzout = xyzout.cpu().numpy()
     tree3=KDTree(xyzout)
     dist, idx = tree3.query(xyzout, k=k)
     avg=np.mean(dist,axis=1)
     avgtotal=np.mean(dist)
    # idx=np.where(avg<avgtotal*1.5)[0]
     idx = np.where(avg < a*avgtotal )[0]
     xyzout=xyzout[idx,:]
     xyzout=torch.tensor(xyzout, dtype=torch.float32).to(device)
     return xyzout


def tensor_remove_outliers(xyzout, k, a):
    # 获取设备信息
    device = xyzout.device

    # 计算点之间的成对欧氏距离
    n = xyzout.size(0)
    xyzout_expanded = xyzout.unsqueeze(1).expand(n, n, -1)  # (n, 1, 3) -> (n, n, 3)
    dist_sq = torch.sum((xyzout_expanded - xyzout.unsqueeze(0).expand(n, n, -1)) ** 2, dim=2)  # (n, n)

    # 找到每个点的k个最近邻
    dist_sq_topk, idx_topk = torch.topk(dist_sq, k, dim=1, largest=False)  # (n, k)

    # 计算到k个最近邻的平均距离
    avg = torch.mean(dist_sq_topk, dim=1)  # (n,)
    avgtotal = torch.mean(avg)  # 标量

    # 根据阈值过滤点
    idx = torch.where(avg < a * avgtotal)[0]  # (m,) m <= n
    xyzout_filtered = xyzout[idx, :]  # (m, 3)

    return xyzout_filtered

#nearest_neighbor_interpolation
def batch_nearest_neighbor_interpolation(input_clouds):
    """
    对批量输入点云进行最近邻插值，

    参数:
    input_clouds (torch.Tensor): 输入点云批次，形状为 (B, N, C)，
                                 其中 B 是批次大小，N 是点的数量，C 是特征数。

    返回:
    torch.Tensor: 插值后的点云批次，形状为 (B, M, C)，其中 M 是插值后点的数量。
    """
    B, N, C = input_clouds.size()
    if torch.isnan(input_clouds).any() or torch.isinf(input_clouds).any():
        raise ValueError("Input clouds contain NaN or Inf values.")
    # 扩展维度以便进行广播计算
    # input_clouds: (B, N, C) -> (B, N, 1, C)
    # input_clouds_t: (B, N, C) -> (B, 1, N, C)
    input_clouds_expanded = input_clouds.unsqueeze(2)  # (B, N, 1, C)
    input_clouds_transposed = input_clouds.unsqueeze(1)  # (B, 1, N, C)

    # 计算批次内所有点之间的欧几里得距离的平方
    # diff: (B, N, N, C)
    # diff = input_clouds_expanded - input_clouds_transposed
    # dist_squared = torch.sum(diff ** 2, dim=-1)  # (B, N, N)
    dist_squared = torch.norm(input_clouds_expanded - input_clouds_transposed, dim=-1)

    # 设置对角线为无穷大，以避免自身最近邻
    diag_mask = torch.eye(N, dtype=torch.bool, device=input_clouds.device)
    diag_mask = diag_mask.unsqueeze(0)
    diag_mask = diag_mask.expand(B, -1, -1)  # (B, N, N)
    dist_squared[diag_mask] = float('inf')

    # 找到最近邻的索引
    nearest_idx = torch.argmin(dist_squared, dim=-1)  # (B, N)

    # 使用高级索引获取最近邻点
    # nearest_points: (B, N, C)
    nearest_points = torch.gather(input_clouds, dim=1, index=nearest_idx.unsqueeze(-1).expand(-1, -1, C))

    # 计算插值点
    # interpolated_points: (B, N, C)
    interpolated_points = (input_clouds + nearest_points) / 2.0

    combined_points = torch.cat([input_clouds, interpolated_points], dim=1)

    # 如果需要，只返回插值后的点云
    return combined_points   #(B,2*N,C)




def compute_nearest_neighbor_distances_batch(points):
    # points: Tensor of shape (B, N, C)
    B, N, C = points.size()

    # Expand points to (B, N, 1, C) and (B, 1, N, C) to compute pairwise differences
    points_expanded_1 = points.unsqueeze(2) .to(memory_format=torch.channels_last)  # (B, N, 1, C)
    points_expanded_2 = points.unsqueeze(1) .to(memory_format=torch.channels_last)  # (B, 1, N, C)
    # Compute pairwise squared Euclidean distances
    dist_sq = torch.sum((points_expanded_1 - points_expanded_2) ** 2, dim=3)  # (B, N, N)

    # Set the diagonal (self-distances) to a large value to ignore them
    mask = torch.eye(N, device=points.device).expand(B, N, N)  # (B, N, N)
    dist_sq = dist_sq + mask * 1e6  # Set self-distance to a large value

    # Find the minimum distance for each point in each batch (excluding self)
    nearest_neighbor_distances, _ = torch.min(dist_sq, dim=2)  # (B, N)

    return nearest_neighbor_distances


def ball_query_centroid1(points, radius):
    """
    Compute the centroid of points within a ball for each point in the point cloud using tensor operations.

    Parameters:
    - points: Tensor of shape (B, C, N) where B is batch size, C is number of dimensions, N is number of points.
    - radius: Float, radius of the ball for querying.

    Returns:
    - centroids: Tensor of shape (B, C, N) containing centroids for each ball.
    """
    B, C, N = points.shape

    # 确保 radii 的形状为 (B, 1)
    assert radius.shape == (B, 1), "radii 的形状必须为 (B, 1)"

    # 将 points 转换为 (B, N, C) 以方便计算
      # 形状: (B, N, C)

    # 计算批次内所有点之间的成对距离
    diff = points.unsqueeze(2) - points.unsqueeze(1)  # 形状: (B, N, N, C)
    distances = torch.norm(diff, dim=3)  # 形状: (B, N, N)

    # 扩展半径以进行广播，形状从 (B, 1) 扩展到 (B, N, N)
    radii = radius.unsqueeze(2)  # 形状: (B, 1, N) -> (B, N, 1) 再广播到 (B, N, N)

    # 创建掩码以识别在半径范围内的点
    within_radius_mask = distances <=10*radii  # 形状: (B, N, N)
   # within_radius_mask = (distances >= radii) & (distances <= 1.3 * radii)
    # 计算每个球内点的总和和计数
    sum_points = torch.einsum('bnc,bni->bci', points, within_radius_mask.float()).permute(0,2,1) # 形状: (B, C, N)
    count_points = within_radius_mask.sum(dim=2,keepdim=True).float()  # 形状: (B, N, 1)

    # 通过扩展维度确保广播，计算质心
    count_points_clamped = torch.clamp(count_points, min=1.0)

    # Compute centroids
    centroids = sum_points / count_points_clamped

    # 将质心转换回 (B, C, N)


    return centroids

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def ball_query_centroid(points, radius):
    """
    Compute the centroid of points within a ball for each point in the point cloud using tensor operations.

    Parameters:
    - points: Tensor of shape (B, C, N) where B is batch size, C is number of dimensions, N is number of points.
    - radius: Float, radius of the ball for querying.

    Returns:
    - centroids: Tensor of shape (B, C, N) containing centroids for each ball.
    """
    B, N, C = points.shape

    # 确保 radii 的形状为 (B, 1)
    assert radius.shape == (B, 1), "radii 的形状必须为 (B, 1)"

    # 将 points 转换为 (B, N, C) 以方便计算
      # 形状: (B, N, C)

    # 计算批次内所有点之间的成对距离
    diff = points.unsqueeze(2) - points.unsqueeze(1)  # 形状: (B, N, N, C)
    distances = torch.norm(diff,p=2, dim=3)  # 形状: (B, N, N)
   # print(distances)
    diag_mask = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    distances[diag_mask] = float('inf')
   # nearest_idx = torch.argmin(distances, dim=-1)  # (B, N)(B, N)
   # nearest_distances = torch.gather(distances, dim=-1, index=nearest_idx.unsqueeze(-1)).squeeze(-1)
    # 扩展半径以进行广播，形状从 (B, 1) 扩展到 (B, N, N)
    radii = radius.unsqueeze(2)  # 形状: (B, 1, N) -> (B, N, 1) 再广播到 (B, N, N)

    # 创建掩码以识别在半径范围内的点
    # within_radius_mask = distances <= 4*radii  # 形状: (B, N, N)
    within_radius_mask =distances <=10* radii

   # within_radius_mask = (distances >= radii) & (distances <= 1.3 * radii)

    # true_counts = within_radius_mask.sum(dim=2)
   # print(true_counts)
    # 计算每个球内点的总和和计数
    sum_points = torch.einsum('bnc,bni->bci', points, within_radius_mask.float()) # 形状: (B, C, N)
    count_points = within_radius_mask.sum(dim=2,keepdim=True).float()  # 形状: (B, N, 1)

    # 通过扩展维度确保广播，计算质心
    count_points_clamped = torch.clamp(count_points, min=1.0)

    # Compute centroids
    centroids = sum_points / count_points_clamped.view(B, 1, N)

    # 将质心转换回 (B, C, N)


    return centroids.permute(0,2,1)




def Calculated_centroid1(points):
    device = points.device  # 确保在GPU上运行
    B, N, C = points.shape  # 批处理大小，点云数量和每个点的维度

    # 计算每个批次中点的最近邻距离
    nearest_neighbor_distances = compute_nearest_neighbor_distances_batch(points).to(device)

    # 计算每个批次的平均距离 d
    d_batch = torch.sum(nearest_neighbor_distances, dim=1) / N  # (B,)
    avg_distances = d_batch.unsqueeze(1)

    # 使用平均距离 d 作为半径执行球查询
    centroid=ball_query_centroid1(points,avg_distances).to(device)
   # centroid=ball_query_centroid1(points,avg_distances).to(device)

    return centroid

def batch_uniform_interpolation(input_clouds):
    """
    对批量输入点云进行均匀插值。

    参数:
    input_clouds (torch.Tensor): 输入点云批次，形状为 (B, N, C)，
                                 其中 B 是批次大小，N 是点的数量，C 是特征数。

    返回:
    torch.Tensor: 插值后的点云批次，形状为 (B, 2*N, C)，其中 2*N 是插值后点的数量。
    """
    B, N, C = input_clouds.shape
    if torch.isnan(input_clouds).any() or torch.isinf(input_clouds).any():
        raise ValueError("Input clouds contain NaN or Inf values.")

    # 扩展输入点云以便进行批处理最近邻搜索
    expanded_cloud = input_clouds.unsqueeze(2).expand(B, N, N, C)
    diff = expanded_cloud - expanded_cloud.transpose(1, 2)
    dist_squared = torch.sum(diff ** 2, dim=-1)

    # 设置对角线为无穷大，以避免自身最近邻
    diag_mask = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
    dist_squared[diag_mask] = float('inf')
    # 获取两个最近邻的索引
    nearest_idx = torch.topk(dist_squared, k=2, largest=False).indices

    # 获取最近邻点
    nearest_points = torch.gather(input_clouds.unsqueeze(2).expand(B, N, N, C), dim=1,
                                  index=nearest_idx.unsqueeze(-1).expand(-1, -1, -1, C))

    # 计算均匀插值点 (简单平均)
    interpolated_points = nearest_points.mean(dim=2)

    # 将原始点和插值点组合在一起
  #  combined_points = torch.cat([input_clouds, interpolated_points], dim=1)

    return  interpolated_points



def save_xyz_file(numpy_array ,xyz_dir):
    num_points = numpy_array.shape[0]
    with open(xyz_dir, 'w') as f:
        for i in range(num_points):
            line = "%f %f %f\n" % (numpy_array[i, 0], numpy_array[i, 1], numpy_array[i, 2])
            f.write(line)
    return
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
    return np.array(points,dtype=np.float32)
if __name__ =="__main__":
   # point_cloud = torch.rand(5, 160, 3).cuda()
    preprocessor =upmesh11(r=4)
    preprocessor.cuda()  # 将模型移动到 GPU
    # #
   # downsampled_point_cloud = preprocessor(point_cloud)
    # print(downsampled_point_cloud.shape)
   #  file_path ='../poisson_256_00003.xyz'
   #  point_cloud_data = read_point_cloud(file_path)
   #
   #  point=np.array(point_cloud_data)
   #  point_cloud = torch.tensor(point_cloud_data, dtype=torch.float32).unsqueeze(0)
   # # point_cloud =point_cloud.permute(0,2,1)
   #  down_point =index_points(point_cloud,down)
   #  point_cloud = torch.rand(100, 3)
   #
   #  d =Calculated_centroid(point_cloud)
   #  preprocessor = upmesh1(r=4)
   #  preprocessor.cuda()  # 将模型移动到 GPU
    file_name = 'poisson_256_00003.xyz'
    file_name1='poisson_1024_00003.xyz'
    directory = '../network'
    file_path = os.path.join(directory, file_name)
    file_path1 = os.path.join(directory, file_name1)
    gt_point =parse_xyz_file(file_path1)
    dataset1 = dataset(file_path)
    dataloader = DataLoader(dataset1, batch_size=1, shuffle=False)
    for batch in dataloader:
       point_tensor = batch
      # print(point_tensor)
       point_tensor =point_tensor.cuda()
       downsampled_point_cloud = preprocessor(point_tensor).cuda()
      # print(downsampled_point_cloud)
       pred =downsampled_point_cloud.cpu().numpy()

       preds =downsampled_point_cloud.data.cpu().numpy()[0]

     #  print(preds)
       #保存为.xyz文件
      # preds = pred.data.cpu().numpy()[0]
       save_file = 'xxx.xyz'

       save_xyz_file(preds, save_file)

      # gt =point_tensor.cpu().numpy()[0]
       print(gt_point.shape)
       pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred)
       gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt_point)
       cd_forward, cd_backward = chamfer_distance1(torch.from_numpy(gt_tensor).cuda(),
                                                   torch.from_numpy(pred_tensor).cuda())
       cd_forward_value = cd_forward[0, :].cpu().numpy()
       cd_backward_value = cd_backward[0, :].cpu().numpy()
       md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
       hd_value = np.max(np.amax(cd_forward_value, axis=0) + np.amax(cd_backward_value, axis=0))
       cd_forward_value = np.mean(cd_forward_value)
       cd_backward_value = np.mean(cd_backward_value)
       CD = cd_forward_value + cd_backward_value
       HD = hd_value
    print(CD)
    print(HD)
#  fig = plt.figure()
   #  ax = fig.add_subplot(121, projection='3d')
   #  ax.scatter(down_point[:, 0], down_point[:, 1], down_point[:, 2])
   #  ax.set_xlabel('x')
   #  ax.set_ylabel('y')
   #  ax.set_zlabel('z')
   #  ax.set_title('3D Point Cloud 1')
   #  ax.view_init(elev=30, azim=45)
   #
   #  ax2 = fig.add_subplot(122, projection='3d')  # 122表示1行2列的第2个子图
   #  ax2.scatter(point[:, 0], point[:, 1], point[:, 2], c='r', marker='o', s=10)
   #  ax2.set_xlabel('X2')
   #  ax2.set_ylabel('Y2')
   #  ax2.set_zlabel('Z2')
   #  ax2.set_title('3D Point Cloud 2')
   #  ax2.view_init(elev=30, azim=135)
   #  plt.show()
  #  print(down_point.shape)


