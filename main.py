import argparse
import time
import csv
import logging
import random
import torch
import pathlib, shutil
import torch.utils.data as data
import numpy as np

import os
import sys

sys.path.append('../')
sys.path.append(os.getcwd())
from tqdm import tqdm
from glob import glob
from collections import defaultdict, OrderedDict
from datetime import datetime
from math import log

from network import operations

from network.SPUDMI import SPUNet as CRNet

from network.model_loss import CD_dist
from network.model import Model
from utils import pc_util
from data.data import H5Dataset
from network.model_loss import UniformLoss as UniformLoss_ps
from uniformLoss.loss import Loss as UniformLoss

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train',
                    help='train, test or eval [default: train]')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--name', default='release',
                    help="experiment name, prepended to log_dir")
parser.add_argument('--log_dir', default='./model/demo/release',
                    help='Log dir [default: log]')
parser.add_argument('--result_dir', default="./model/test/result",
                    help='result directory')
parser.add_argument('--ckpt',
                    help='model to restore from')
parser.add_argument('--num_point', type=int, default='256',
                    help='Input Point Number [default: 256]')
parser.add_argument('--num_shape_point', type=int, default='256',
                    help="Number of points per shape")
parser.add_argument('--up_ratio', type=int, default=4,
                    help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=28,
                    help='Batch Size during training，default 24')
parser.add_argument('--noise', type=float, default=0.,
                    help='Add random Gaussian noise to the test data to verify robustness, generally set to 0.01 to 0.2')

parser.add_argument('--h5_data',
                    default='./dataset/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5',
                    help='h5 file for training')

parser.add_argument('--test_data', default='./dataset/PU1K/test/input_2048/input_2048',
                    help='test data path')
parser.add_argument('--gt_path', default='./dataset/PU1K/test/input_2048/gt_8192',
                    help='test gt data path')

parser.add_argument('--decay_iter', type=int, default=50000)
parser.add_argument('--lr_init', type=float, default=0.001)
parser.add_argument('--restore_epoch', type=int, default=0)
parser.add_argument('--save_step', type=int, default=5,
                    help='save_step during training')
parser.add_argument('--print_step', type=int, default=800,
                    help='print_step during training')
parser.add_argument('--patch_num_ratio', type=float, default=3)
parser.add_argument('--jitter', action="store_true",
                    help="jitter augmentation")
parser.add_argument('--jitter_sigma', type=float,
                    default=0.005, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float,
                    default=0.01, help="jitter augmentation")
parser.add_argument('--random_seed', type=int, default=42)

args = parser.parse_args()
print('=' * 10, 'params', '=' * 10)
print(args)
print('=' * 10, 'params', '=' * 10)
DEVICE = torch.device('cuda', args.gpu)  # cuda:0 or cuda:1
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)


def train(conf):
    save_model_dir = conf.log_dir
    best_cd = 30
    logdir = os.path.join(save_model_dir, 'tensorboard')
    writer_tensorboard = SummaryWriter(logdir)
    result_dir = os.path.join(save_model_dir, 'result')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(save_model_dir, 'train.log')),
                                  logging.StreamHandler(sys.stdout)])

    net = CRNet(conf.up_ratio)
    total_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(
        "===number of trainable parameters in upsampler: {:.4f} K === ".format(float(total_trainable_parameters / 1e3)))

    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.train()
    model = Model(net, "train", conf)


    dataset = H5Dataset(h5_path=conf.h5_data, num_shape_point=conf.num_shape_point, num_patch_point=conf.num_point,
                        batch_size=conf.batch_size, up_ratio=conf.up_ratio, jitter=conf.jitter)
    dataloader = data.DataLoader(dataset, batch_size=conf.batch_size, pin_memory=True, num_workers=4, shuffle=True,
                                 drop_last=True)

    print("===================================================")
    if conf.restore_epoch:
        model_name = os.path.join(save_model_dir, 'model_{:d}.pth'.format(conf.restore_epoch))
        logging.info("Loadding model from {} ".format(model_name))
        net.module.load_state_dict(torch.load(model_name)['net_state_dict'])
        start_epoch = conf.restore_epoch
    else:
        start_epoch = 1

    for epoch in range(start_epoch, conf.max_epoch + 1):
        if epoch % conf.save_step == 0:
            path = os.path.join(save_model_dir, 'model_{:d}.pth'.format(epoch))
            torch.save({'net_state_dict': net.module.state_dict()}, path)
            net.eval()
            print("--------------------------------------------")
            target_folder = test(conf.result_dir, net)
            net.train()
            cd, hd = online_evaluation(target_folder, conf.gt_path, save_model_dir,
                                       writer_tensorboard=writer_tensorboard, step=epoch)
            logging.info("epoch: {:d}, CD: {:.3f}, HD: {:.3f} ".format(epoch, cd, hd))

            if cd < best_cd:
                path = os.path.join(save_model_dir, 'model_best.pth')
                torch.save({'net_state_dict': net.module.state_dict()}, path)
                best_cd = cd
        for i, examples in enumerate(dataloader):
            total_batch = i + (epoch - 1) * len(dataloader)
            input_pc, label_pc, radius = examples
            input_pc = input_pc.cuda()
            print(input_pc.shape)  #(32,3,256)
            # =======Unsupervision training
            label_pc = input_pc.clone().cuda()
            radius = radius.cuda()

            model.set_input(input_pc,radius,label_pc=label_pc)
           # print(type(model.set_input(input_pc,radius,mesh_point,label_pc=label_pc)))  #

            loss, lr = model.optimize(total_batch, epoch)
            if i % conf.print_step == 0:
                # logging.info(
                #     "epoch: %d, iteration: %d, Lr: %.6f, Loss_1: %.6f, Loss_2: %.6f, Loss_3: %.6f. losses: %s" % (
                #         epoch, i, lr, loss[0], loss[1], loss[2], str(loss)))
                logging.info(
                    "epoch: %d, iteration: %d, Lr: %.6f, Loss_1: %.6f, Loss_2: %.6f, losses: %s" % (
                        epoch, i, lr, loss[0], loss[1], str(loss)))
    path = os.path.join(save_model_dir, 'model.pth')
    torch.save({'net_state_dict': net.module.state_dict()}, path)


def pc_prediction(net, input_pc, patch_num_ratio=3):
    """
    upsample patches of a point cloud
    :param
        input_pc        1x3xN
        patch_num_ratio int, impacts number of patches and overlapping
    :return
        input_list      list of [3xM]
        up_point_list   list of [3xMr]
    """
    num_patches = int(input_pc.shape[2] / args.num_point * patch_num_ratio)  # [1,3,N],or [1, 3, 2048]
    start = time.time()
    idx, seeds = operations.fps_subsample(input_pc, num_patches, NCHW=True)
    print("number of patches: %d" % seeds.shape[-1])

    input_list = []
    up_point_list = []
    patches, _, _ = operations.group_knn(args.num_point, seeds, input_pc,
                                         NCHW=True)

    patch_time = 0.
   # print(num_patches)
    for k in range(num_patches):
        patch = patches[:, :, k, :]
      #  print(patch.shape)   #[1,3,256]
        patch, centroid, furthest_distance = operations.normalize_point_batch(
            patch, NCHW=True)
        start = time.time()
        up_point = net.forward(patch.detach().contiguous() )
       # print(up_point.shape)   #1,1024,3
        end = time.time()
        patch_time += end - start
       # print("-------------")
        if (up_point.shape[0] != 1):
            up_point = torch.cat(
                torch.split(up_point, 1, dim=0), dim=2)
            _, up_point = operations.fps_subsample(
                up_point, args.num_point)

        if up_point.size(1) != 3:  #检查第二维是不是3
            assert (up_point.size(2) == 3), "ChamferLoss is implemented for 3D points"
            up_point = up_point.transpose(2, 1).contiguous()
        up_point = up_point * furthest_distance + centroid
        input_list.append(patch)

        up_point_list.append(up_point)


   # print("结束")
    return input_list, up_point_list, patch_time / num_patches


def test(result_dir, net=None, shape_count=2048):

    test_files = glob(args.test_data + '/**/*.xyz', recursive=True)
    # print('args.test_data:',args.test_data)
    # print('test_files:',test_files)
    # print('test_files:',test_files)
    total_time = 0.
    for point_path in test_files:
        folder = os.path.basename(os.path.dirname(point_path))
        target_folder = os.path.join(result_dir, folder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        out_path = os.path.join(target_folder, point_path.split('/')[-1][
                                               :-4] + '.xyz')
        data = pc_util.load(point_path, shape_count)
        if args.noise != 0.:
            noise = np.random.normal(0, args.noise, data.shape)
            # noise = np.random.normal(0, 1, data.shape)*args.noise
            data += noise
        data = data[np.newaxis, ...]
        data, centroid, furthest_distance = pc_util.normalize_point_cloud(data)

        # transpose to NCHW format
        data = torch.from_numpy(data).transpose(2, 1).to(device=DEVICE).float()
        start = time.time()
        with torch.no_grad():
            input_pc_list, pred_pc_list, avg_patch_time = pc_prediction(
                net, data, patch_num_ratio=args.patch_num_ratio)
        pred_pc = torch.cat(pred_pc_list, dim=-1)
        end = time.time()
        total_time += avg_patch_time

        _, pred_pc = operations.fps_subsample(
            pred_pc, shape_count * args.up_ratio, NCHW=True)


        pred_pc = pred_pc.transpose(2, 1).cpu().numpy()

        pred_pc = (pred_pc * furthest_distance) + centroid
        pred_pc = pred_pc[0, ...]

        np.savetxt(out_path[:-4] + '.xyz', pred_pc, fmt='%.6f')

   #print('Average Inference Time: {} ms'.format(total_time / len(test_files) * 1000.))
    #print("走到这")
    return target_folder


def offline_test(result_dir):

    net = CRNet(args.up_ratio)

    total_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(
        "===number of trainable parameters in upsampler: {:.4f} K === ".format(float(total_trainable_parameters / 1e3)))
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    net.module.load_state_dict(torch.load(args.ckpt)['net_state_dict'])
    net.eval()
    target_folder = test(result_dir, net, shape_count=args.num_shape_point)

    return target_folder


def evaluation(target_folder, gt_folder, save_path):

    precentages = np.array([0.008, 0.012])
    fieldnames = ["name", "CD", "hausdorff", "p2f avg", "p2f std"]
    # fieldnames += ["uniform_%d" % d for d in range(precentages.shape[0])]
    print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))
    UniformLoss_ = UniformLoss()
    avg_uniform_value = [0, 0, 0, 0, 0]
    gt_paths = glob(os.path.join(gt_folder, '*.xyz'))
    gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
    cd_dist_compute = CD_dist()
    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    counter = 0
    print('target_folder:',target_folder)
    pred_paths = glob(os.path.join(target_folder, "*.xyz"))
    gt_pred_pairs = []
    print('pred_paths:',pred_paths)
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert (ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))
    torch.set_printoptions(precision=6)

    global_p2f = []
    with open(os.path.join(save_path, "evaluation.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()
        for gt_path, pred_path in gt_pred_pairs:
            row = {}
            gt = pc_utils.load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]
            gt, centroid, furthest_distance = pc_util.normalize_point_cloud(gt)

            gt = torch.from_numpy(gt).to(device=DEVICE)

            pred = pc_utils.load(pred_path)
            pred = pred[:, :3]
            row["name"] = os.path.basename(pred_path)
            pred = pred[np.newaxis, ...]

            pred, centroid, furthest_distance = pc_util.normalize_point_cloud(pred)
            pred = torch.from_numpy(pred).to(device=DEVICE)

            cd_forward_value, cd_backward_value = cd_dist_compute(pred, gt)
            cd_forward_value = np.array(cd_forward_value.cpu())
            cd_backward_value = np.array(cd_backward_value.cpu())

            md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
            hd_value = np.max(np.amax(cd_forward_value, axis=1)[0] + np.amax(cd_backward_value, axis=1)[0])
            cd_backward_value = np.mean(cd_backward_value)
            cd_forward_value = np.mean(cd_forward_value)
            row["CD"] = cd_forward_value + cd_backward_value
            row["hausdorff"] = hd_value
            avg_md_forward_value += cd_forward_value
            avg_md_backward_value += cd_backward_value
            avg_hd_value += hd_value

            uloss_1 = UniformLoss_.get_uniform_loss_one(pred, p=0.004)
            uloss_2 = UniformLoss_.get_uniform_loss_one(pred, p=0.006)
            uloss_3 = UniformLoss_.get_uniform_loss_one(pred, p=0.008)
            uloss_4 = UniformLoss_.get_uniform_loss_one(pred, p=0.01)
            uloss_5 = UniformLoss_.get_uniform_loss_one(pred, p=0.012)
            avg_uniform_value[0], avg_uniform_value[1], avg_uniform_value[2], avg_uniform_value[3], avg_uniform_value[
                4] = \
                avg_uniform_value[0] + uloss_1, avg_uniform_value[1] + uloss_2, avg_uniform_value[2] + uloss_3, \
                avg_uniform_value[3] + uloss_4, avg_uniform_value[4] + uloss_5

            if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
                point2mesh_distance = pc_util.load(pred_path[:-4] + "_point2mesh_distance.xyz")
                if point2mesh_distance.size == 0:
                    continue
                point2mesh_distance = point2mesh_distance[:, 3]
                row["p2f avg"] = np.nanmean(point2mesh_distance)
                row["p2f std"] = np.nanstd(point2mesh_distance)
                global_p2f.append(point2mesh_distance)

            writer.writerow(row)
            counter += 1

        row = OrderedDict()
        avg_md_forward_value /= counter
        avg_md_backward_value /= counter
        avg_hd_value /= counter
        avg_cd_value = avg_md_forward_value + avg_md_backward_value
        row["CD"] = avg_cd_value
        row["hausdorff"] = avg_hd_value
        # row["uniform"] = avg_uniform_value/counter

        if global_p2f:
            global_p2fs = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2fs)
            std_p2f = np.nanstd(global_p2fs)
            row["p2f avg"] = mean_p2f
            row["p2f std"] = std_p2f

        writer.writerow(row)
        row = OrderedDict()
        row["CD (1e-3)"] = avg_cd_value * 1000.
        row["hausdorff (1e-3)"] = avg_hd_value * 1000.
        if global_p2f:
            global_p2fs = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2fs)
            std_p2f = np.nanstd(global_p2fs)
            row["p2f avg (1e-3)"] = mean_p2f * 1000.
            row["p2f std (1e-3)"] = std_p2f * 1000.

        print(" | ".join(["{:>15.8f}".format(d) for d in row.values()]))

        print(
            '========uniform loss (1e-3)========\n p=0.004 |       p=0.006 |       p=0.008 |       p=0.010 |       p=0.012')
        print(' | '.join(["{:>15.8f}".format(v.cpu().item() * 1000 / counter) for v in avg_uniform_value]))

        with open(os.path.join(save_path, "finalresult.text"), "w") as text:
            print(row, file=text)


def online_evaluation(PRED_DIR, GT_DIR, save_path, writer_tensorboard=None, step=None):

    uniformloss = UniformLoss_ps(loss_name='uniform', alpha=1)
    avg_uniform_value = 0

    precentages = np.array([0.008, 0.012])

    fieldnames = ["name", "CD", "hausdorff", "p2f avg", "p2f std"]
    fieldnames += ["uniform_%d" % d for d in range(precentages.shape[0])]

    gt_paths = glob(os.path.join(GT_DIR, '*.xyz'))

    gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
    # gt = load(gt_paths[0])[:, :3]
    cd_dist_compute = CD_dist()
    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    avg_emd_value = 0
    counter = 0
    pred_paths = glob(os.path.join(PRED_DIR, "*.xyz"))
    gt_pred_pairs = []
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert (ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))
    torch.set_printoptions(precision=6)

    with open(os.path.join(save_path, "evaluation.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()

        pred_list = []
        for gt_path, pred_path in gt_pred_pairs:
            row = {}
            gt = pc_util.load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]
            gt, centroid, furthest_distance = pc_util.normalize_point_cloud(gt)

            gt = torch.from_numpy(gt).to(device=DEVICE)

            pred = pc_util.load(pred_path)
            pred = pred[:, :3]
            pred_list.append(pred)
            row["name"] = os.path.basename(pred_path)
            pred = pred[np.newaxis, ...]

            pred, centroid, furthest_distance = pc_util.normalize_point_cloud(pred)

            pred = torch.from_numpy(pred).to(device=DEVICE)

            cd_forward_value, cd_backward_value = cd_dist_compute(pred, gt)
            cd_forward_value = np.array(cd_forward_value.cpu())
            cd_backward_value = np.array(cd_backward_value.cpu())

            md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
            hd_value = np.max(np.amax(cd_forward_value, axis=1) + np.amax(cd_backward_value, axis=1))
            cd_backward_value = np.mean(cd_backward_value)
            cd_forward_value = np.mean(cd_forward_value)

            row["CD"] = cd_forward_value + cd_backward_value
            row["hausdorff"] = hd_value
            avg_md_forward_value += cd_forward_value
            avg_md_backward_value += cd_backward_value
            avg_hd_value += hd_value

            uloss = uniformloss(pred)
            avg_uniform_value += uloss

            writer.writerow(row)
            counter += 1

        pred_all = np.array(pred_list)
        print('pred_all.shape:', pred_all.shape)
        writer_tensorboard.add_mesh('pred', vertices=pred_all[:18, :, :], global_step=step)
        row = OrderedDict()
        avg_md_forward_value /= counter
        avg_md_backward_value /= counter
        avg_hd_value /= counter
        avg_emd_value /= counter
        avg_cd_value = avg_md_forward_value + avg_md_backward_value
        row["CD"] = avg_cd_value
        row["hausdorff"] = avg_hd_value
        row["EMD"] = avg_emd_value

        avg_uniform_value = avg_uniform_value / counter
        print('uniform loss:', avg_uniform_value)

        writer.writerow(row)

        row = OrderedDict()
        row["CD (1e-3)"] = avg_cd_value * 1000.
        row["hausdorff (1e-3)"] = avg_hd_value * 1000.
        return avg_cd_value * 1000., avg_hd_value * 1000.


def generate_exp_directory(conf):
    """
    Helper function to create checkpoint folder. We save
    model checkpoints using the provided model directory
    but we add a sub-folder for each separate experiment:
    """
    experiment_string = conf.name

    pathlib.Path(conf.log_dir).mkdir(parents=True, exist_ok=True)
    if conf.phase == 'train':
        conf.log_dir = os.path.join(conf.log_dir, experiment_string)
        conf.code_dir = os.path.join(conf.log_dir, "code")
        pathlib.Path(conf.code_dir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    result_path = args.result_dir
    if args.phase == "test":
        print(args.ckpt)
        out_path = offline_test(result_path)
    elif args.phase == "eval":
        print('args.test_data:',args.test_data)
        folder = args.test_data.split('/')[-2]
        out_path = os.path.join(result_path, folder)
        gt_path = args.gt_path
        evaluation(out_path, gt_path, os.path.dirname(args.ckpt))
    elif args.phase == "train":
        generate_exp_directory(args)
        train(args)

