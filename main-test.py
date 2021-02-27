import os
import sys
import argparse
import time
import random
import copy
import shutil
import scipy.io as sio
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

import load_datasets, models, utils


dataset_cands = ['yaleB', 'PIE', 'FEI']

parser = argparse.ArgumentParser(description='Commands')

# run_time
parser.add_argument('-r', '--run-time', type=str, default='2020-08-08', metavar='DIR',
                    help='programs run time')
# dir
parser.add_argument('-s', '--save-dir', type=str, default='res', metavar='DIR',
                    help='save directory')
parser.add_argument('--data-dir', type=str, default='data', metavar='DIR',
                    help='data directory')
# dataset
parser.add_argument('-d', '--dataset', type=str, default='yaleB', metavar='DSET',
                    choices=dataset_cands,
                    help='dataset: ' + ' | '.join(dataset_cands) + ' (default: yaleB)')
                    
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-t', '--task-size', type=int, nargs='+', default=[8, 10], metavar='N+',
                    help='number of initial classes and incremental classes (default: 8 10)')
# training
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to init-train (default: 50)')
parser.add_argument('--ldpc-len', type=int, default=256, metavar='N',
                    help='ldpc length (default: 256)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='R',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lrd', type=float, default=0.1, metavar='R',
                    help='learning rate decay (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='R',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='R',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--schedule', type=int, nargs='+', default=[25], metavar='N+',
                    help='when to decay SGD learning rate to init train (default: 25)')
# architecture
parser.add_argument('-a', '--net', type=str, default='resnet_pre', metavar='NET',
                    help='CNN architecture (default: resnet_pre)')
parser.add_argument('--drop', '--dropout', default=0.5, type=float, metavar='PROB',
                    help='Dropout ratio')
# device
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--dataset-seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default='0', type=str, metavar='N,',
                    help='argument for CUDA_VISIBLE_DEVICES (default: 0)')

# parse
args = parser.parse_args()
# deterministic random numbers
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
cpu_device = torch.device('cpu')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if 'win' in sys.platform:
    split_flag = '\\'
else:
    split_flag = '/'

def save_pred_results(pred_result, pred_result_path):
    sio.savemat(
        pred_result_path, {
            'preOut' : pred_result})

def test_init(test_loader, model, test_results_dir):
    pred_result_path = os.path.join(test_results_dir, test_results_dir.split('/')[-2]+'.mat')
    model.to(device)
    model.eval()

    pbar  = tqdm(test_loader, desc='test', ascii=True, ncols=80)
    i = 0
    for data, _, src in pbar:
        src = src.numpy()
        data_size = len(src)
        img_idx = np.arange(data_size) 
        data = data.to(device)
        with torch.no_grad():
            if i == 0:
                pred_result = model(data)
                pred_result_arr = pred_result[0].cpu().numpy()
                for task_idx in range(1, len(pred_result)):
                    pred_result_arr = np.vstack((pred_result_arr, pred_result[task_idx].cpu().numpy()))
            else:
                pred_result_item = model(data)
                pred_result_item_arr = pred_result_item[0].cpu().numpy()
                for task_idx in range(1, len(pred_result_item)):
                    pred_result_item_arr = np.vstack((pred_result_item_arr, pred_result_item[task_idx].cpu().numpy()))
                
                pred_result_arr = np.vstack((pred_result_arr, pred_result_item_arr))
        del data
        torch.cuda.empty_cache()
        i += 1

    save_pred_results(pred_result_arr, pred_result_path)


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(args)

    cur_folder = os.getcwd()
    
    label_len = 256
    run_time = split_flag + args.run_time + '-'
    sub_model_results_dir = run_time + args.net + '-codes{}-epochs{}-seed{}-dataset-seed{}'.format(label_len, args.epochs, args.seed, args.dataset_seed)
    sub_model_dir = os.path.join(sub_model_results_dir, 'models')
    sub_test_result_dir = os.path.join(sub_model_results_dir, 'test')

    model_dir = os.path.join(cur_folder, args.dataset+sub_model_dir)
    test_result_dir = os.path.join(cur_folder, args.dataset+sub_test_result_dir)
    
    print('model_dir: ', model_dir)
    print('test_result_dir: ', test_result_dir)
    
    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)

 
    if args.dataset == 'yaleB':
        num_classes = 38
    elif args.dataset == 'PIE':
        num_classes = 68
    elif args.dataset == 'FEI':
        num_classes = 200

    dataset_dir = os.path.join(cur_folder, 'test_images/'+args.dataset+'_dataset_seed'+str(args.dataset_seed))
    test_dataset_dir = os.path.join(dataset_dir, 'crop_249_test_resize_224')
    class_order = np.load('split/'+args.dataset.upper() + '_class_order_{}.npy'.format(num_classes))[args.seed].tolist()
    tasks = []
    p = 0
    t = 0
    while p < num_classes:
        inc = args.task_size[1] if p > 0 else args.task_size[0]
        tasks.append(class_order[p:p+inc])
        p += inc
    print('seed = ', args.seed, 'dataset_seed = ', args.dataset_seed, 'tasks: ', tasks)

    # save str
    base_str = args.dataset # dataset name default: 'yaleB'
    base_str += '_seed{seed:d}'.format(seed=args.seed)
    new_str = '{base_str}'.format(base_str=base_str)
   
    print('base_str: ', base_str)
    print('new_str: ', new_str)

    # data loader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    isPreNet = False
    if 'pre' in args.net:
        isPreNet = True

    # test dataset
    test_test_transform = utils.get_transform(dataset=args.dataset, phase='test', isPreNet=isPreNet)
    test_test_dataset = load_datasets.FaceDataSet(
        dataset_name=args.dataset, dataset_dir=test_dataset_dir,
        transform=test_test_transform,tasks=tasks,
        network=args.net, seed=args.seed)
    test_test_dataset.load_new_task(t=0) # 每一个task分别取当前task及其之前的每次task的测试数据集
    test_test_loader = DataLoader(test_test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs)
   
    # find model and load
    model_path = '{}/model_{}_{:d}.pt'.format(model_dir, base_str, t)
    print(model_path)
    model = utils.create_model(args.net, label_len, args.gpu)
    model.add_classes() # 初始模型，只需要一个classifier
    
    
    if os.path.isfile(model_path):
        """
        载入上次保存的模型参数,传递给当前模型
            """
        model_stored_state_dict = torch.load(model_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_stored_state_dict)
        else:
            model.load_state_dict(model_stored_state_dict)
        print('model parameters loaded')
        test_init(test_test_loader, model, test_result_dir)
    else:
        print('没有找到模型')
