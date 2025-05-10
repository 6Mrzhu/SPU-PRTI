from collections import defaultdict

from model_loss import ChamferLoss, OffsetLoss, UniformLoss
from torch.optim.lr_scheduler import ExponentialLR
from uniformLoss.loss import Loss
import torch
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


from SPUDMI import SPUNet
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
    distance = torch.ones(B, N).to(device) * 1e10
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
class Model(object):
    def __init__(self, net, phase, opt, writer_tensorboard=None):
        self.net = net
        self.phase = phase
        self.writer_tensorboard = writer_tensorboard
        if self.phase == 'train':
            self.error_log = defaultdict(int)
            self.chamfer_criteria = ChamferLoss()
            self.uniformloss = UniformLoss(loss_name='uniform', alpha=1)
            self.offsetLoss = OffsetLoss()

            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr_init,
                                              betas=(0.9, 0.999))
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.7)
            self.decay_step = opt.decay_iter
        self.step = 0

    def set_input(self, input_pc, radius, label_pc=None):
        """`
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.radius = radius
        self.R=4
        self.input = input_pc.detach()
        B, C, N = input_pc.shape
        downsamplenum = int(N / self.R)  # 

        far_point = farthest_point_sample(xyz=input_pc.permute(0, 2, 1), npoint=downsamplenum)
        input_point = index_points(input_pc.permute(0, 2, 1), far_point)
        self.input = input_point.detach().permute(0, 2, 1).contiguous().cuda()  #B,C,N
        if label_pc is not None:
            self.gt = label_pc.detach().permute(0, 2, 1).cuda()  #(B,N,C)
        else:
            self.gt = None

        return self.input, self.gt
    def forward(self):
        if self.gt is not None:
            self.predicted, self.gt = self.net(self.input,self.gt)
        else:
            self.predicted = self.net(self.input)

    def get_lr(self, optimizer):
        """Get the current learning rate from optimizer.
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def optimize(self, steps=None, epoch=None):
        """
        run forward and backward, apply gradients
        """
        self.optimizer.zero_grad()
        self.net.train()
        self.forward()


        P1, P2,P3,P4 = self.predicted

        gt_downsample_idx = furthest_point_sample(self.gt.contiguous(), int(P1.shape[1]))
        gt_downsample = gather_operation(self.gt.contiguous(), gt_downsample_idx)
        gt_downsample = gt_downsample.permute(0, 2, 1).contiguous()
        cd_1 = Loss().get_cd_loss(P1,gt_downsample)
        cd_2 =Loss().get_cd_loss(P2,gt_downsample)
        cd_3 = Loss().get_cd_loss(P3, self.gt)
        cd_4 = Loss().get_cd_loss(P4, self.gt)


        alpha = 0.2
        uniform_1 = self.uniformloss(P1) * alpha
        uniform_2 = self.uniformloss(P2) * alpha
        uniform_3 = self.uniformloss(P3) * alpha
        uniform_4 = self.uniformloss(P4) * alpha

        loss1, loss2, loss3, loss4 =cd_1 , uniform_2, cd_3 , uniform_4
        self.gs = self.net.module.get_gs()
        self.offsets = self.net.module.get_offsets()


        loss = loss1 + loss2 + loss3 + loss4
        losses = [loss1.item(), loss2.item(), loss3.item(), loss4.item()]


        loss.backward()
        self.optimizer.step()

        if steps % self.decay_step == 0 and steps != 0:
            self.lr_scheduler.step()
        lr = self.get_lr(self.optimizer)
        return losses, lr
if __name__ =="__main__":
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SPUNet(up_ratio=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
  #  net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.train()
    model = Model(net=net, phase='train', opt=type('Options', (object,), {'lr_init': 0.001, 'decay_iter': 1000}))
    point_cloud = torch.rand(1,3, 512).cuda()
    point_cloud1 =point_cloud.cpu().numpy()
    data_radius = np.ones(shape=(len(point_cloud)))
    data_radius = torch.tensor(data_radius, dtype=torch.float32).to(device)
    label_pc=point_cloud.clone()
    model.set_input(point_cloud, data_radius, label_pc=label_pc)
    total_batch = 10
    epoch =2
    loss, lr = model.optimize(total_batch, epoch)

    print("Losses:", loss)
    print("Learning Rate:", lr)
    print(model)
